# -*- coding: utf-8 -*-
# file: td_lstm.py


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class BERT_TD_LSTM(nn.Module):
    def __init__(self, bert, opt):
        """

        :param bert: 初始化的bert模型
        :param opt:
        """
        super(BERT_TD_LSTM, self).__init__()
        # embedding_matrix [dict_words, embedding_dim]; self.embed  [dict_words, embedding_dim]
        self.bert = bert
        # 使用的动态LSTM
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        """
        :param inputs: 输入是
        inputs[0]   text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
        inputs[1]   text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
        :return:
        """
        x_l, x_r = inputs[0], inputs[1]
        #统计长度
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l_context, x_l_output = self.bert(x_l)
        x_r_context, x_r_output = self.bert(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l_context, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r_context, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
