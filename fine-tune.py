# -*- coding: utf-8 -*-
import sys
import os
bert_path = os.path.join(os.getcwd(), 'bert')
sys.path.append(bert_path)


import json
import numpy as np
import time
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from scipy.spatial.distance import cosine

from albert.albert_total import get_albert_total
from torch import nn
import torch.nn.functional as F 
from model import BigModel, BertTextNet
from optim import Optim
from adam import AdamWeightDecayOptimizer
from data import RawDataLoader

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = './data/txt'
data_path = './data'

train_data = RawDataLoader(dataset_path, 20)
print_every = 50

# 现成的BERT用来做句子划分
bert_model = BertTextNet()

# 组合的大模型
model = BigModel(device)
model = model.to(device)

# 模型优化器使用BERT默认的
optimizer = Optim(model_size=768, factor=1, warmup=10000,\
    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

batch_acm = 0
acc1_acm = 0.
acc2_acm = 0.
loss_acm = 0.
stoa_acc = 0.

while True:
    model.train()
    # 使用数据集，每次读取20条句子
    for sentences in train_data:
        batch_acm += 1

        tokens, segments, input_masks = [], [], []
        for sentence in sentences:
            text = "[CLS] {} [SEP]".format(sentence)

            tokenized_text = bert_model.tokenizer.tokenize(text)  # 用tokenizer对句子分词
            indexed_tokens = bert_model.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度


        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens).to(device)
        segments_tensors = torch.tensor(segments).to(device)
        input_masks_tensors = torch.tensor(input_masks).to(device)
        #print(tokens_tensor[0])
        #print(segments_tensors.size(0),input_masks_tensors.size())
        model.zero_grad()
        loss, acc1, acc2 = model(tokens_tensor, segments_tensors, input_masks_tensors)
        loss_acm += loss.item()
        acc1_acm += acc1
        acc2_acm += acc2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if batch_acm%print_every == 0:
            print('epoch %d, batch_acm %d, loss %.3f, easy_acc %.3f, hard_acc %.3f'\
                %(train_data.epoch_id, batch_acm, loss_acm/print_every, acc2_acm/print_every, acc1_acm/print_every), flush=True)
            if acc1_acm/print_every > stoa_acc:
                stoa_acc = acc1_acm/print_every
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_batch_%dacc_%.3f'%(data_path, train_data.epoch_id, batch_acm, stoa_acc))
            loss_acm, acc1_acm, acc2_acm = 0., 0., 0.



