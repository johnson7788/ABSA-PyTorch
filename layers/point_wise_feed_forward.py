# -*- coding: utf-8 -*-
# file: point_wise_feed_forward.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        """
        Point-wise卷积变换（PCT）方法，公式7
        :param d_hid:
        :param d_inner_hid:
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        # 做了2层卷积 w_1 -->w_2
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        进行2次卷积操作
        :param x:  shape [batch_size, seq_len, hidden_dim]
        :return:
        """
        # x.transpose(1, 2) shape  [batch_size,hidden_dim, seq_len]
        # self.w_1(x.transpose(1, 2)) --> [batch_size, d_inner_hid, seq_len]
        output = self.relu(self.w_1(x.transpose(1, 2)))
        # [batch_size,d_inner_hid, seq_len] -->  [batch_size,hidden_dim, seq_len] --> [batch_size, seq_len, hidden_dim]
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output
