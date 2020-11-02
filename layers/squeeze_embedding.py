# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import numpy as np

class SqueezeEmbedding(nn.Module):
    """
    挤压序列嵌入长度为批次中最长的
    由默认全部一样的默认序列长度，挤压成这个batch中最大的序列长度
    """
    def __init__(self, batch_first=True):
        """
        初始化
        :param batch_first:
        """
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        由默认全部一样的默认序列长度，挤压成这个batch中最大的序列长度,
        :param x: sequence embedding vectors  未排序的x [batch_size, seq_length]  eg: [16,80]
        :param x_len: numpy/tensor list   x的每个句子的长度 eg:[55,33,30,29,...]
        :return:  x维度 [16,55], 把多余的padding的0挤压出去了
        """
        """排序，从句子长度最长到最短"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        # 长度从大到小排列
        x_len = x_len[x_sort_idx]
        # x也调整成这个排列
        x = x[x_sort_idx]
        """pack 将一个填充过的变长序列 压紧"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """unpack: out, 这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        # unpack回来的序列x, out.shape [batch_size, max_seq_len]
        out = out[0]  #
        """unsort"""
        out = out[x_unsort_idx]
        return out
