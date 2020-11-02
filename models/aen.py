# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_GloVe(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEN_GloVe, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out


class AEN_BERT(nn.Module):
    def __init__(self, bert, opt):
        """
        注意力编码器网络, Attentional Encoder Network for Targeted Sentiment Classiﬁcation
        :param bert:
        :param opt:
        """
        super(AEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        # attn_k和 attn_q的初始化
        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        # 初始化ffn_c, PCT层
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        # 目标特定的注意力层初始化
        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        # 最终输出层定义
        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        # context.shape, target.shape  [batch_size, seq_length],
        # context --> text_raw_bert_indices --> ("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        # target --> aspect_bert_indices --> ("[CLS] " + aspect + " [SEP]")
        context, target = inputs[0], inputs[1]
        # context_len, target_len -->batch_size ,  统计句话和aspect的长度, eg context_len tensor([23, 55, 32, 20, 29, 17, 55, 18, 23, 16, 25, 20, 18, 24,  9, 27])
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        # 第一部分 embedding
        # 挤压序列嵌入长度为批次中最长的, context 【batch_size, max_length_of_this_batch]
        context = self.squeeze_embedding(context, context_len)
        # bert输出的维度，[batch_size, max_length_of_this_batch, embedding_dim] eg: torch.Size([16, 55, 768])
        context, _ = self.bert(context)
        context = self.dropout(context)
        # 同理context,
        target = self.squeeze_embedding(target, target_len)
        # 同理context,[batch_size, max_length_of_this_batch, embedding_dim]
        target, _ = self.bert(target)
        target = self.dropout(target)
        # 第二部分，通过MHA和PCT，计算hc,ht
        # Intra-MHA部分，计算context之间的注意力, MHA:Multi_head_attention, hc shape [batch_size, seq_len, project_layer_hidden_size]
        hc, _ = self.attn_k(context, context)
        # PCT部分
        hc = self.ffn_c(hc)
        # 同理，进行context和target之间的MHA和PCT
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        # 第三部分, Target-specific Attention Layer， 2个PCT的输出进行计算attention, ht.shape [batch_size, aspect_seq_len, hidden_size] eg: torch.Size([16, 7, 300])
        # hc.shape  [batch_size, context_seq_len, hidden_size] eg: torch.Size([16, 55, 300])
        s1, _ = self.attn_s1(hc, ht)
        # context_len [batch_size] 每个context的长度组成的batch
        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        # 第四部分, 平均池化后输出
        # 对hc，ht，和s1分别在batch维度上计算均值, hc是context间，ht是 context和apsect之间, s1是hc和ht之间
        #  对hc的维度1做sum --> [batch_size, context_seq_len, hidden_size] -->[batch_size, hidden_size]
        # context_len.view(context_len.size(0), 1) --> [batch_size, 1]
        # hc_mean shape [batch_size, hidden_size], eg:torch.Size([16, 300])
        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        # hc_mean shape [batch_size, hidden_size], eg:torch.Size([16, 300])
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        # hc_mean shape [batch_size, hidden_size], eg:torch.Size([16, 300])
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))
        # 把 hc_mean, s1_mean, ht_mean  拼接 -->[batch_size, hidden_size * 3]
        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        # [batch_size, class_num]
        out = self.dense(x)
        return out
