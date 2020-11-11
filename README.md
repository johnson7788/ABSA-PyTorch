# ABSA-PyTorch

## Aspect Based Sentiment Analysis, PyTorch Implementations.
基于方面的情感分析，使用PyTorch实现。

```
![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
[![Gitter](https://badges.gitter.im/ABSA-PyTorch/community.svg)](https://gitter.im/ABSA-PyTorch/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
```

## Requirement

* pytorch >= 0.4.0
* numpy >= 1.13.3
* sklearn
* python 3.6 / 3.7
* transformers

To install requirements, run `pip install -r requirements.txt`.

## 数据集格式
$T$ 代表句子中的短语，1表示POS，0表示NEU，-1表示NEG
```buildoutcfg
$T$ is super fast , around anywhere from 35 seconds to 1 minute .
Boot time
1
$T$ would not fix the problem unless I bought your plan for $ 150 plus .
tech support
-1
No $T$ is included .
installation disk (DVD)
0
```

## 模型训练参数讲解
```buildoutcfg
train.py
  --model_name MODEL_NAME
                        要使用的模型，模型在models目录下,例如 aen_bert，bert_spc，lcf_bert等
  --dataset DATASET     数据集名称
  --optimizer OPTIMIZER
                        模型训练优化器，默认adam
  --initializer INITIALIZER
                        参数初始化方法xavier_normal_,orthogonal_
  --learning_rate LEARNING_RATE
                        学习率BERT用 5e-5, 2e-5，其它用模型用 1e-3
  --dropout DROPOUT     默认的droput 0.1
  --l2reg L2REG         weight_decay ,l2正则系数
  --num_epoch NUM_EPOCH
                        非BERT类模型，训练的epoch需要多一些
  --batch_size BATCH_SIZE
                        BERT模型请使用如下batch_size 16, 32, 64
  --log_step LOG_STEP   每多少step进行日志记录
  --embed_dim EMBED_DIM
                        词嵌入时embedding的维度，默认300
  --hidden_dim HIDDEN_DIM
                        训练时隐藏层维度，默认300
  --bert_dim BERT_DIM   bert dim，bert的词向量维度，默认768
  --pretrained_bert_name PRETRAINED_BERT_NAME
                        使用的预训练模型，使用的预训练模型，默认bert-base-chinese
  --pretrained_bert_cache_dir PRETRAINED_BERT_CACHE_DIR
                        使用的预训练模型缓存的目录，默认
  --embedding_file EMBEDDING_FILE
                        如果不使用BERT，那么自定义的预训练的词向量文件的位置
  --max_seq_len MAX_SEQ_LEN
                        最大序列长度
  --polarities_dim POLARITIES_DIM
                        类别维度，分几类,默认POS，NEU，NEG
  --hops HOPS           多少hop设置
  --device DEVICE       e.g. cuda:0
  --seed SEED           用于重现，随机数种子
  --valset_ratio VALSET_RATIO
                        训练集拆分出验证的比例, 在0和1之间设置比例以验证,如果为0，用测试集代替验证集
  --recreate_caches     默认False，是否重新生成数据处理的cache文件
  --local_context_focus LOCAL_CONTEXT_FOCUS
                        本地上下文焦点模式，cdw或cdm
  --SRD SRD             语义相对距离，请参阅LCF-BERT模型的论文
```

* 对于基于非BERT的模型,
[GloVe pre-trained word vectors](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) 是必须的, 请参考 [data_utils.py](./data_utils.py) 更多细节.
http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.42B.300d.zip
下载  glove.42B.300d.zip，解压缩 glove.42B.300d.txt，放在项目根目录

对比golve的中文词向量, 或者参考utils中的train_word2vec函数，训练自己的词向量
https://github.com/Embedding/Chinese-Word-Vectors

## 使用方式

### 训练

```sh
python train.py --model_name bert_spc --dataset restaurant
python train.py --model_name aen_bert --dataset cosmetics --max_seq_len 200 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --batch_size 16 --num_epoch 5
python train.py --model_name bert_spc --dataset cosmetics --max_seq_len 200 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --batch_size 16 --num_epoch 5
python train.py --model_name lcf_bert --dataset cosmetics --max_seq_len 200 --valset_ratio 0.1 --pretrained_bert_name bert-base-chinese --pretrained_bert_cache_dir model_cache --batch_size 16 --num_epoch 3

#使用自定义的bert_td_lstm
python train.py --model_name bert_td_lstm --dataset cosmetics_as --do_train --valset_ratio 0.1 --learning_rate 1e-3 --max_seq_len 70 --batch_size 16 --num_epoch 15 --embed_dim 768
```


* 所有实现的模型都在  [models directory](./models/).
* 查看 [train.py](./train.py) 了解更多训练参数
* 参考 [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) k折交叉验证。

### 推理

* 参考 [infer_example.py](./infer_example.py) 用于基于非BERT的模型.
* 参考 [infer_example_bert_models.py](./infer_example_bert_models.py) 用于基于BERT的模型.

### 提示

* 对于基于非BERT的模型，训练过程不是很稳定。
* 基于BERT的模型对小数据集的超参数（尤其是学习率）更敏感，请参阅 [this issue](https://github.com/songyouwei/ABSA-PyTorch/issues/27).
* 为了释放BERT的真正性能，必须对特定任务进行微调。


## Reviews / Surveys 论文

Qiu, Xipeng, et al. "Pre-trained Models for Natural Language Processing: A Survey." arXiv preprint arXiv:2003.08271 (2020). [[pdf]](https://arxiv.org/pdf/2003.08271)

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)


## BERT-based models

### BERT-ADA ([official](https://github.com/deepopinion/domain-adapted-atsc))

Rietzler, Alexander, et al. "Adapt or get left behind: Domain adaptation through bert language model finetuning for aspect-target sentiment classification." arXiv preprint arXiv:1908.11860 (2019). [[pdf](https://arxiv.org/pdf/1908.11860)]

### BERR-PT ([official](https://github.com/howardhsu/BERT-for-RRC-ABSA))

Xu, Hu, et al. "Bert post-training for review reading comprehension and aspect-based sentiment analysis." arXiv preprint arXiv:1904.02232 (2019). [[pdf](https://arxiv.org/pdf/1904.02232)]

### ABSA-BERT-pair ([official](https://github.com/HSLCY/ABSA-BERT-pair))

Sun, Chi, Luyao Huang, and Xipeng Qiu. "Utilizing bert for aspect-based sentiment analysis via constructing auxiliary sentence." arXiv preprint arXiv:1903.09588 (2019). [[pdf](https://arxiv.org/pdf/1903.09588.pdf)]

### LCF-BERT ([lcf_bert.py](./models/lcf_bert.py)) ([official](https://github.com/yangheng95/LCF-ABSA))

Zeng Biqing, Yang Heng, et al. "LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification." Applied Sciences. 2019, 9, 3389. [[pdf]](https://www.mdpi.com/2076-3417/9/16/3389/pdf)

### AEN-BERT ([aen.py](./models/aen.py))

Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1902.09314 (2019). [[pdf]](https://arxiv.org/pdf/1902.09314.pdf)

### BERT for Sentence Pair Classification ([bert_spc.py](./models/bert_spc.py))

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). [[pdf]](https://arxiv.org/pdf/1810.04805.pdf)


## Non-BERT-based models

### MGAN ([mgan.py](./models/mgan.py))

Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)

### AOA ([aoa.py](./models/aoa.py))

Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)

### TNet ([tnet_lf.py](./models/tnet_lf.py)) ([official](https://github.com/lixin4ever/TNet))

Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)

### Cabasc ([cabasc.py](./models/cabasc.py))

Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

### RAM ([ram.py](./models/ram.py))

Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)

### MemNet ([memnet.py](./models/memnet.py)) ([official](https://drive.google.com/open?id=1Hc886aivHmIzwlawapzbpRdTfPoTyi1U))

Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)

### IAN ([ian.py](./models/ian.py))

Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)

### ATAE-LSTM ([atae_lstm.py](./models/atae_lstm.py))

Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.

### TD-LSTM ([td_lstm.py](./models/td_lstm.py), [tc_lstm.py](./models/tc_lstm.py)) ([official](https://drive.google.com/open?id=17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6))

Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)

### LSTM ([lstm.py](./models/lstm.py))

Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780. [[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)]


## Contributors

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/AlbertoPaz"><img src="https://avatars2.githubusercontent.com/u/36967362?v=4" width="100px;" alt=""/><br /><sub><b>Alberto Paz</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=AlbertoPaz" title="Code">💻</a></td>
    <td align="center"><a href="http://taojiang0923@gmail.com"><img src="https://avatars0.githubusercontent.com/u/37891032?v=4" width="100px;" alt=""/><br /><sub><b>jiangtao </b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=jiangtaojy" title="Code">💻</a></td>
    <td align="center"><a href="https://genezc.github.io"><img src="https://avatars0.githubusercontent.com/u/24239326?v=4" width="100px;" alt=""/><br /><sub><b>WhereIsMyHead</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=GeneZC" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/songyouwei"><img src="https://avatars1.githubusercontent.com/u/2573291?v=4" width="100px;" alt=""/><br /><sub><b>songyouwei</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=songyouwei" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/yangheng95"><img src="https://avatars2.githubusercontent.com/u/51735130?v=4" width="100px;" alt=""/><br /><sub><b>YangHeng</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=yangheng95" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/rmarcacini"><img src="https://avatars0.githubusercontent.com/u/40037976?v=4" width="100px;" alt=""/><br /><sub><b>rmarcacini</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=rmarcacini" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ZhangYikaii"><img src="https://avatars1.githubusercontent.com/u/46623714?v=4" width="100px;" alt=""/><br /><sub><b>Yikai Zhang</b></sub></a><br /><a href="https://github.com/songyouwei/ABSA-PyTorch/commits?author=ZhangYikaii" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Licence

MIT
