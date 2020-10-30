# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        """
        :param bert: 初始化的bert模型
        :param opt: argparse参数列表
        """
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        """
        :param inputs: 一个batch的features列表
        :return:
        """
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
