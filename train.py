# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT, AEN_GloVe
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        """
        初始化模型和数据预处理，并token化
        :param opt: argparse的参数
        """
        self.opt = opt
        #是否是bert类模型，使用bert类模型初始化， 非BERT类使用GloVe
        if 'bert' in opt.model_name:
            #初始化tokenizer
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name, cache_dir=opt.pretrained_bert_cache_dir)
            # 加载BERT模型
            bert = BertModel.from_pretrained(opt.pretrained_bert_name, cache_dir=opt.pretrained_bert_cache_dir)
            # 然后把BERT模型和opt参数传入自定义模型，进行进一步处理
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            # 自定义tokenizer，生成id2word，word2idx
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            #返回所有单词的词嵌入 [word_nums, embedding_dimesion]
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            # 加载模型
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        # 加载训练集
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer, recreate_caches=opt.recreate_caches)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer, recreate_caches=opt.recreate_caches)
        #如果valset_ratio为0，测试集代替验证集
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset
        # 检查cuda的内存
        if opt.device.type == 'cuda':
            logger.info('cuda 可用内存: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        """
        打印模型的参数数量和argparse的参数，n_trainable_params可训练参数 ，n_nontrainable_params不可训练参数
        :return:
        """
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        #打印所有argparse的参数
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        """
        参数初始化，对于BERT模型的参数，不初始化，对于一维参数，使用均匀分布初始化，其它使用initializers定义的初始化
        :return:
        """
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        """
        训练模型
        :param criterion: 损失函数 eg：交叉熵损失
        :param optimizer: 优化器 eg： Adam
        :param train_data_loader:
        :param val_data_loader:
        :return: 返回效果最好的模型的路径
        """
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                # 根据模型需要取出特定的features
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # outputs shape [batch_size, class_num]
                outputs = self.model(inputs)
                #取出这个batch的ground_truth情感标签
                targets = sample_batched['polarity'].to(self.opt.device)
                #通过交叉熵计算损失 outputs: [batch_size, class_num]  targets:[class_num] -->loss
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                #累加训练正确的个数
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                #记录到日志
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('第{}个step的loss: {:.4f}, acc: {:.4f}'.format(global_step,train_loss, train_acc))
            #每个epoch都去验证一次
            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> global_step: {}, val_acc: {:.4f}, val_f1: {:.4f}'.format(global_step, val_acc, val_f1))
            #如果模型准确率提高，那么保存此模型
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_step{2}_val_acc{3}'.format(self.opt.model_name, self.opt.dataset, global_step, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        """
        模型评估
        :param data_loader: 数据集
        :return: 准确率和f1
        """
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        """
        损失和优化器
        :return:
        """
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        # 封装成 DataLoader，
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        #初始化参数
        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        #  加载模型并评估
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str, help='要使用的模型，模型在models目录下')
    parser.add_argument('--dataset', default='laptop', type=str, help='数据集 twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help='参数初始化方法xavier_normal_,orthogonal_')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率BERT用 5e-5, 2e-5，其它用模型用 1e-3')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float, help='weight_decay ,l2正则系数')
    parser.add_argument('--num_epoch', default=10, type=int, help='非BERT类模型，请使用较多epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='BERT模型请使用如下batch_size 16, 32, 64')
    parser.add_argument('--log_step', default=5, type=int,help='每多少step进行日志记录')
    parser.add_argument('--embed_dim', default=300, type=int, help='embedding的维度')
    parser.add_argument('--hidden_dim', default=300, type=int, help='隐藏层维度')
    parser.add_argument('--bert_dim', default=768, type=int, help='bert dim')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str, help='使用的预训练模型')
    parser.add_argument('--pretrained_bert_cache_dir', default=None, type=str, help='使用的预训练模型缓存的目录')
    parser.add_argument('--max_seq_len', default=80, type=int, help='最大序列长度')
    parser.add_argument('--polarities_dim', default=3, type=int, help='类别维度，分几类,默认POS，NEU，NEG')
    parser.add_argument('--hops', default=3, type=int,help='多少hop设置')
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='用于重现，随机数种子')
    parser.add_argument('--valset_ratio', default=0, type=float, help='训练集拆分出验证的比例, 在0和1之间设置比例以验证,如果为0，用测试集代替验证集')
    parser.add_argument('--recreate_caches', action='store_true', help='默认False，是否重新生成数据处理的cache文件')
    # 以下参数仅对lcf-bert模型有效
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='本地上下文焦点模式，cdw或cdm')
    parser.add_argument('--SRD', default=3, type=int, help='语义相对距离，请参阅LCF-BERT模型的论文')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'aen_glove': AEN_GloVe,
        'lcf_bert': LCF_BERT,
        # LCF-BERT模型的默认超参数如下：
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    #数据文件检索
    dataset_files = {
        'twitter': {
            'train': './datasets/twitter/train.raw',
            'test': './datasets/twitter/test.raw'
        },
        'restaurant': {
            'train': './datasets/restaurant/Restaurants_Train.xml.seg',
            'test': './datasets/restaurant/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/laptop/Laptops_Train.xml.seg',
            'test': './datasets/laptop/Laptops_Test_Gold.xml.seg'
        },
        'cosmetics': {
            'train': './datasets/cosmetics/train.txt',
            'test': './datasets/cosmetics/test.txt'
        }
    }
    # 使用哪种特征的文件，我们对数据进行了各种预处理，分别满足不同模型的数据要求, input columns
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tc_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    #参数初始化化方式
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    #所有的优化器
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    # model的class, 都放到opt参数中，备用
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    # cpu还是GPU
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    #日志文件设置
    logdir = "logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    full_log_file = os.path.join(logdir,log_file)
    logger.addHandler(logging.FileHandler(full_log_file))

    # 构建模型
    ins = Instructor(opt)
    # 运行
    ins.run()


if __name__ == '__main__':
    main()
