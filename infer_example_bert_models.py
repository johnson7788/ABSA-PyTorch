# -*- coding: utf-8 -*-
# file: infer_example_bert_models.py
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from models.lcf_bert import LCF_BERT
from models.aen import AEN_BERT
from models.bert_spc import BERT_SPC
from transformers import BertModel
from data_utils import Tokenizer4Bert
import argparse


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def prepare_data(text_left, aspect, text_right, tokenizer):
    """
    处理数据集
    :param text_left:
    :param aspect:
    :param text_right:
    :param tokenizer:
    :return:
    """
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    # 做成tokenizer
    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)            
    aspect_indices = tokenizer.text_to_sequence(aspect)
    # aspect_len长度
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str, help='要使用的模型，模型在models目录下')
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # 推理模型
    model_classes = {
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT
    }
    # set your trained models here
    state_dict_paths = {
        'lcf_bert': 'state_dict/lcf_bert_laptop_val_acc0.2492',
        'bert_spc': 'state_dict/bert_spc_restaurant_val_acc0.3',
        'aen_bert': 'state_dict/aen_bert_restaurant_val_acc0.8'
    }
    sentiment2id = {
        "NEG": -1,
        "NEU": 0,
        "POS": 1
    }
    id2sentiment = {value:key for key,value in sentiment2id.items()}

    opt = get_parameters()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #加载tokenizer和预训练模型
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    #初始化自定义模型, eg: aen_bert
    model = model_classes[opt.model_name](bert, opt).to(opt.device)
    #加载训练好的模型
    print('开始加载模型 {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(state_dict_paths[opt.model_name], map_location=opt.device),strict=False)
    model.eval()
    torch.autograd.set_grad_enabled(False)
    #例如输入如下
    # input:  这个小地方有可爱的室内装饰和负担得起的城市价格
    # input: This little place has a cute interior decor and affordable city prices.
    # text_left = This little place has a cute 
    # aspect = interior decor
    # text_right = and affordable city prices.
    
    text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data('This little place has a cute', 'interior decor', 'and affordable city prices.', tokenizer)
    
    # 转换成tensor 格式
    text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
    bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
    text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
    aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
    if 'lcf' in opt.model_name:
        inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices]
    elif 'aen' in opt.model_name:
        inputs = [text_raw_bert_indices, aspect_bert_indices]
    elif 'spc' in opt.model_name:
        inputs = [text_bert_indices, bert_segments_ids]
    outputs = model(inputs)
    t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
    print('t_probs = ', t_probs)
    sentiment_id = t_probs.argmax(axis=-1) - 1
    sentiment = id2sentiment[sentiment_id[0]]
    print(f"aspect sentiment = {sentiment_id}, {sentiment}")

