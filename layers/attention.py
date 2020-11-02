# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism 注意力机制
        :param embed_dim: 嵌入维度; eg: 768
        :param hidden_dim: 隐藏维度; eg: 96
        :param out_dim: 输出维度   eg:300
        :param n_head: num of head (Multi-Head Attention)  head的数量  eg:8
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)   eg: mlp， bi_linear， dot_product, scaled_dot_product
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        # Q和K的计算
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        #attention的投影层计算
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """权重初始化"""
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        """
        计算MHA， Mutil Head attention
        :param k: 默认输入维度 [batch_size, seq_length, embedding_dim]
        :param q:  [batch_size, seq_length, embedding_dim]
        :return:
        """
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        # mb_size 是batch_size
        mb_size = k.shape[0]  # ?
        # k_len和q_len是seq_len
        k_len = k.shape[1]
        q_len = q.shape[1]
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        # self.w_k(k)的维度[batch-size,seq_len, n_head * hidden_dim]
        # kx 维度 [batch-size,seq_len, n_head, hidden_dim]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        #【heads, batch_size, seq_len, hidden_dim】kx.permute(2, 0, 1, 3).contiguous().shape --> torch.Size([8, 16, 55, 96])
        # view(-1, k_len, self.hidden_dim) 合并head和bath_size维度 [heads * batch_size, seq_len,hidden_dim]
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        # qx同 kx维度
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        #
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            # torch.unsqueeze(kx, dim=1) 对kx扩充一个维度, [heads * batch_size, seq_len,hidden_dim]--> [heads * batch_size, 1, seq_len,hidden_dim]
            # expand(-1, q_len, -1, -1)， 把维度2扩充到seq_len维度, [heads * batch_size, seq_len, seq_len,hidden_dim]
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            # torch.unsqueeze(qx, dim=2) -->  [heads * batch_size, seq_len,hidden_dim]  --> [heads * batch_size,seq_len, 1 ,hidden_dim]
            # 把维度3扩充到seq_len维度, [heads * batch_size, seq_len, seq_len,hidden_dim]
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            # kq.shape (n_head*batch_size, q_len, k_len, hidden_dim*2)
            kq = torch.cat((kxx, qxx), dim=-1)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            # score shape [heads * batch_size, q_len, seq_len]
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        # score shape  [heads * batch_size, q_len, seq_len]
        score = F.softmax(score, dim=-1)
        # bmm， 批次的batch1和batch2内的矩阵进行批矩阵乘操作 [heads * batch_size, q_len, hidden_dim]
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        # torch.split(output, mb_size, dim=0) --> 拆分出8个head，每个维度[batch_size, q_len,hidden_dim], 然后拼接所有n_head
        # output的维度 [batch_size, q_len,hidden_dim *n_head]
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        # [batch_size, seq_len, proj_layer_hidden_dim] eg: torch.Size([16, 55, 300])
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
