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


from conf.config import bert_config_path, bert_model_path, bert_vocab_path # bert_base_chinese
from conf.config import albert_config_path, albert_model_path, albert_vocab_path # albert_base

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_init(model_path, device):
    ckpt = torch.load(model_path, map_location='cpu')
    big_model = BigModel(device)
    big_model.load_state_dict(ckpt['model'])
    big_model = big_model.to(device)
    big_model.eval()
    return big_model


# 读取数据库中的句子和768-dim的向量
txt_path = './data/txt'
vec_path = './data/vec'
files = os.listdir(txt_path)
sentences = []
vecs = np.zeros((1, 768))

start = time.time()
print('begin to read database!')
for txt_name in files:
    txt_p = os.path.join(txt_path, txt_name)
    with open(txt_p, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
        t_sentences = load_dict['content']
        sentences = sentences + t_sentences

    vec_p = os.path.join(vec_path, txt_name)
    mat = np.loadtxt(vec_p)
    mat = mat.reshape(-1, 768)
    vecs = np.concatenate((vecs, mat), axis=0)
vecs = vecs[1:]
end_database = time.time()
print('all data has been read. time: ', end_database - start)
print('total sentence number: ', len(sentences), ', vector size: ', vecs.shape)

# 当前需要查询的句子，可以是多个句子，list形式
sentence_list=['她猛地站起来，苍白的脸由于愤怒而胀红，双眼充满泪水']


# 载入google默认的BERT来对目标句子做tokenizer
bert_model = BertTextNet()

tokens, segments, input_masks = [], [], []
for sentence in sentence_list:
    text = "[CLS] {} [SEP]".format(sentence)
    tokenized_text = bert_model.tokenizer.tokenize(text)  # 用tokenizer对句子分词
    indexed_tokens = bert_model.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

max_len = max([len(single) for single in tokens])  # 最大的句子长度


for j in range(len(tokens)):
    padding = [0] * (max_len - len(tokens[j])) # 剩余长度做填充。
    tokens[j] += padding
    segments[j] += padding
    input_masks[j] += padding

tokens_tensor = torch.tensor(tokens).to(device)
segments_tensors = torch.tensor(segments).to(device)
input_masks_tensors = torch.tensor(input_masks).to(device)

end_split = time.time()
print('sentences have been split by tokenizer, time: ', end_split - end_database)

# 读取组合的模型CKPT
model_path = './data/CKPT_C'
big_model = model_init(model_path, device)
print('big_model has been loaded')

start_query = time.time()
# 模型输出预测的句子表示 (1, 768)
q_v = big_model.work(tokens_tensor, segments_tensors, input_masks_tensors).cpu().detach().numpy()

# 和数据库中所有的句子做点乘后排序
score = np.sum(q_v * vecs, axis=1) / np.linalg.norm(q_v, axis=1) / np.linalg.norm(vecs, axis=1)
idx = np.argsort(score)[::-1]
end_query = time.time()
# 返回检索到的句子
print('retrieval ended, time: ', end_query - start_query)
print('current sentences: ', sentence_list)
print('1:', sentences[idx[0]])
print('2:', sentences[idx[1]])
print('3:', sentences[idx[2]])
print('4:', sentences[idx[3]])
print('5:', sentences[idx[4]])

