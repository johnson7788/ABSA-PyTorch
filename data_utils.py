# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

def build_tokenizer(fnames, max_seq_len, dat_fname):
    """
    :param fnames:
    :param max_seq_len:
    :param dat_fname:
    :return:
    """
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """
    对序列进行padding和截取
    :param sequence: 序列id，例如 [2021, 1996, 3095, 2001, 2061, 9202, 2000, 2149, 1012]
    :param maxlen: 最大序列长度
    :param dtype: 数据转换成numpy int64格式
    :param padding: post 还是pre
    :param truncating: post 还是pre
    :param value: 默认填充的值，默认用0填充
    :return: [2021 1996 3095 2001 2061 9202 2000 2149 1012    0    0    0    0    0,    0    0    0    0    0    0    0    0    0    0    0    0    0    0,    0    0    0    0    0    0    0    0    0    0    0    0    0    0,    0    0    0    0    0    0    0    0    0    0    0    0    0    0,    0    0    0    0    0    0    0    0    0    0    0    0    0    0,    0    0    0    0    0    0    0    0    0    0]
    """
    #初始化创建maxlen序列长度的numpy 列表，
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        #从前面截断
        trunc = sequence[-maxlen:]
    else:
        #从后面截断
        trunc = sequence[:maxlen]
    #截断后的数据转换成numpy
    trunc = np.asarray(trunc, dtype=dtype)
    #从前还是后面padding
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        """
        :param max_seq_len:  最大序列长度
        :param pretrained_bert_name:  预训练的bert模型名称
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        """
        文本序列tokenizer化
        :param text: 输入的文本序列, eg: but the staff was so horrible to us.
        :param reverse: 反转所有
        :param padding:  pre或 post， 在序列签名padding还是后面padding
        :param truncating: pre或 post，如果序列过长，是从前面截断，还是后面
        :return: 返回处理后的序列
        """
        # tokenize(text) 是按bert进行tokenize，例如'but the staff was so horrible to us .' -->['but', 'the', 'staff', 'was', 'so', 'horrible', 'to', 'us', '.']
        # sequence 变成序列[2021, 1996, 3095, 2001, 2061, 9202, 2000, 2149, 1012]
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        # 是否翻转
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        """
        数据集处理
        :param fname: 训练集或测试集，验证集等的详细文件路径例如 './datasets/restaurant/Restaurants_Train.xml.seg'
        :param tokenizer:  已经初始化的tokenizer
        """
        # 读取所有行，形成一个列表
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        #用于存储处理后的数据
        all_data = []
        # 每3行为一个完整的样本
        for i in range(0, len(lines), 3):
            # lines[i].partition("$T$") 输出 ('But the ', '$T$', ' was so horrible to us .\n')
            # text_left是'$T$'左边部分，text_right是右边部分
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            # aspect行处理
            aspect = lines[i + 1].lower().strip()
            #polarity情感，是0，-1，或1
            polarity = lines[i + 2].strip()
            # 包含aspect的完整文本序列由文本到--> id
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            # 不包含aspect的文本处理，--> id
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            # aspect左边的文本处理
            text_left_indices = tokenizer.text_to_sequence(text_left)
            # aspect左边的文本+aspect处理
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            # 右边序列处理，并且做反转
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            # aspect+右边序列，并且做反转
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            # 单独aspect处理
            aspect_indices = tokenizer.text_to_sequence(aspect)
            # 左边序列的长度
            left_context_len = np.sum(text_left_indices != 0)
            # aspect的长度
            aspect_len = np.sum(aspect_indices != 0)
            # aspect在文本中的位置tensor([2, 2])
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            # 情感转换成正数 -1,0,1 --> 0,1,2 ,  0：NEG， 1：NEU， 2：POS
            polarity = int(polarity) + 1
            # 构造成BERT格式，SEP分隔，sentence1是完全的句子，sentence2是aspect
            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            #
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
