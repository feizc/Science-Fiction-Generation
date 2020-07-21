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
from model import PrefixPredict


from conf.config import bert_config_path, bert_model_path, bert_vocab_path # bert_base_chinese
from conf.config import albert_config_path, albert_model_path, albert_vocab_path # albert_base

import warnings
warnings.filterwarnings("ignore")

def model_init(model_path, device):
    ckpt = torch.load(model_path, map_location='cpu')
    lm_model = PrefixPredict(device)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model

def sentence_prediction(lm_model, device, inp):
    inp = inp.to(device)
    return lm_model.work(inp)

class BertTextNet(nn.Module):
    def __init__(self):
        """
        bert模型。
        """
        super(BertTextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(bert_config_path)
        self.textExtractor = BertModel.from_pretrained(
            bert_model_path, config=modelConfig)
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_path)

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


class BertSeqVec(object):
    def __init__(self, text_net):
        """
        接收一个bert或albert模型，对文本进行向量化。
        :param text_net: bert或albert模型实例。
        """
        self.text_net = text_net
        self.tokenizer = text_net.tokenizer

    def seq2vec(self, text):
        """
        对文本向量化。
        :param text:str，未分词的文本。
        :return:
        """
        text = "[CLS] {} [SEP]".format(text)
        tokens, segments, input_masks = [], [], []

        tokenized_text = self.tokenizer.tokenize(text)  # 用tokenizer对句子分词
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)
        text_hashCodes = self.text_net(tokens_tensor, segments_tensors,
                                       input_masks_tensors)  # text_hashCodes是bert模型的文本特征
        return text_hashCodes[0].detach().numpy()


class AlbertTextNet(BertTextNet):
    def __init__(self):
        """
        albert 文本模型。
        """
        super(AlbertTextNet, self).__init__()
        config, tokenizer, model = get_albert_total(albert_config_path, albert_model_path, albert_vocab_path)
        self.textExtractor = model
        self.tokenizer = tokenizer

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


if __name__ == '__main__':
    device = 0
    # 读取数据库
    txt_path = './data/txt'
    vec_path = './data/vec'
    files = os.listdir(txt_path)
    sentences = []
    vecs = np.zeros((1, 768))

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
    print('all data has been read. ')
    print('total sentence number: ', len(sentences), ', vector size: ', vecs.shape)

    
    # 载入bert模型
    bert_model = BertTextNet()
    seq2vec = BertSeqVec(bert_model)
    print('Bert model loaded.')


    #载入下一句表示的预测模型
    model_path = './data/CKPT'
    lm_model = model_init(model_path, device)
    print('Prefix model loaded.')

    # 给定句子
    query = '时间是够的，要相信联合政府！这我说了多少遍，如果你们还不相信，我们就退一万步说：人类将自豪地去死，因为我们尽了最大的努力！'
    # Bert模型编码
    q_v = seq2vec.seq2vec(query).reshape(1, 768)
    inp = torch.from_numpy(q_v)
    print('Transform sentence to vector sucessfully!!')

    # 对下一句做出预测
    res_vec = sentence_prediction(lm_model, device, inp).cpu().detach().numpy()
    print('Predict next sentence vector sucessfully!')

    # 从数据库中找到最相似的前5句话返回。
    score = np.sum(res_vec * vecs, axis=1) / np.linalg.norm(res_vec, axis=1) / np.linalg.norm(vecs, axis=1)
    idx = np.argsort(score)[::-1]

    print('q:', query)
    print('1:', sentences[idx[0]])
    print('2:', sentences[idx[1]])
    print('3:', sentences[idx[2]])
    print('4:', sentences[idx[3]])
    print('5:', sentences[idx[4]])


