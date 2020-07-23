import torch
from torch import nn
import torch.nn.functional as F 

from utils import gelu, LayerNorm
from transformer import TransformerLayer, SelfAttentionMask

import sys
import os
bert_path = os.path.join(os.getcwd(), 'bert')
sys.path.append(bert_path)


import json
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from albert.albert_total import get_albert_total

from conf.config import bert_config_path, bert_model_path, bert_vocab_path # bert_base_chinese
from conf.config import albert_config_path, albert_model_path, albert_vocab_path # albert_base

class ContrativeLoss(nn.Module):
    def __init__(self, device, factor=1.0):
        super(ContrativeLoss, self).__init__()
        self.device = device
        self.factor = factor

    def accuracy_compute(self, positive_pro, negative_pro1, negative_pro2, negative_pro3, negative_pro4):
        assert len(positive_pro)==len(negative_pro1)==len(negative_pro2)==len(negative_pro3)
        correct1_num = 0
        correct2_num = 0
        for i in range(len(positive_pro)):
            if positive_pro[i] > negative_pro1[i] and positive_pro[i] > negative_pro2[i]:
                correct1_num += 1
            if positive_pro[i] > negative_pro3[i] and positive_pro[i] > negative_pro4[i]:
                correct2_num += 1
        return correct1_num / len(positive_pro), correct2_num/len(positive_pro)

    def forward(self, output, correct, future_hard, previous_hard, future_easy, previous_easy):
        # output = [sentence_num, 768]
        output_norm = torch.norm(output, dim=1)
        positive_pro = torch.sum(output*correct, dim=1) / torch.norm(correct, dim=1) / output_norm
        negative_pro1 = torch.sum(output*future_hard, dim=1) / torch.norm(future_hard, dim=1) / output_norm
        negative_pro2 = torch.sum(output*previous_hard, dim=1) / torch.norm(previous_hard, dim=1) / output_norm
        negative_pro3 = torch.sum(output*future_easy, dim=1) / torch.norm(future_easy, dim=1) / output_norm
        negative_pro4 = torch.sum(output*previous_easy, dim=1) / torch.norm(previous_easy, dim=1) / output_norm
        # probability = [sentence_num, ]
        loss = - torch.log(torch.exp(positive_pro) / (torch.exp(positive_pro)\
            + self.factor * torch.exp(negative_pro1) + self.factor * torch.exp(negative_pro2)+\
            self.factor * torch.exp(negative_pro3) + self.factor * torch.exp(negative_pro4)))
        accuracy1, accuracy2 = self.accuracy_compute(positive_pro, negative_pro1, negative_pro2, negative_pro3, negative_pro4)
        return loss, accuracy1, accuracy2


# input the previous and current sentence hidden from bert (768-dim)
# predict the hidden state of next sentence (768-dim)
# the structure is identical to gpt without embedding
class PrefixPredict(nn.Module):
    def __init__(self, local_rank, input_dim=768, ff_dim=2048, num_heads=8, dropout=0.2, layers=6):
        super(PrefixPredict, self).__init__()
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(input_dim, ff_dim, num_heads, dropout))
        self.one_more = nn.Linear(input_dim, input_dim)
        self.one_more_layer_norm = LayerNorm(input_dim)

        self.attn_mask = SelfAttentionMask(device=local_rank)
        self.loss_fun = ContrativeLoss(device=local_rank)
        self.dropout = dropout
        self.device = local_rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)

    def work(self, inp):
        sentence_num, inp_dim = inp.size()
        self_attn_mask = self.attn_mask(sentence_num)
        x = inp.unsqueeze(1)
        for layer in self.layers:
            x, _, _ = layer(x, self_attn_mask = self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        x = x.squeeze(1)
        return x[-1:]
        # return [1, 768]

    def forward(self, inp, correct, future_hard, previous_hard, future_easy, previous_easy):
        seq_len, input_dim = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        # x = [sentence_num, bsz, input_dim]
        x = inp.unsqueeze(1)
        for layer in self.layers:
            x, _, _ = layer(x, self_attn_mask = self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        x = x.squeeze(1)
        # loss = [sentence_num, ]
        loss, acc1, acc2 = self.loss_fun(x, correct, future_hard, previous_hard, future_easy, previous_easy)

        return x, loss.mean(), acc1, acc2


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
        return text_hashCodes[0]


class BigModel(nn.Module):

    """BERT和Transformer拼成的大模型"""
    def __init__(self, device, prefix_model_path='./data/CKPT'):
        super(BigModel, self).__init__()
        self.device = device
        # 初始化BERT模型
        self.bert_model = BertTextNet()
        # self.seq2vec = BertSeqVec(self.bert_model)
        # 初始化Prefix模型
        self.lm_model = self.model_init(prefix_model_path, device)

    # 读入CKPT来初始化模型。
    def model_init(self, model_path, device):
        ckpt = torch.load(model_path, map_location='cpu')
        # ckpt = torch.load(model_path)
        lm_model = PrefixPredict(device)
        lm_model.load_state_dict(ckpt['model'])
        # lm_model = lm_model.to(device)
        lm_model.train()
        return lm_model

    # 读入表示向量组，生成对应的正样本和负样本
    def sample_generation(self, vectors):
        # vectors = [40, 768]
        inp = vectors[:-1]
        correct = vectors[1:]
        future_hard = torch.cat((vectors[2:], vectors[:1]), 0)
        previous_hard = torch.cat((vectors[-1:], vectors[:-2]), 0)
        future_easy = torch.cat((vectors[6:], vectors[:5]), 0)
        previous_easy = torch.cat((vectors[-5:], vectors[:-6]), 0)
        return inp, correct, future_hard, previous_hard, future_easy, previous_easy

    def work(self, tokens_tensor, segments_tensors, input_masks_tensors):
        # input [sentence_num, max_len]*3
        vectors = torch.randn(1, 768).to(self.device)
        sentence_num, max_len = tokens_tensor.size()
        for i in range(sentence_num):
            output = self.bert_model(tokens_tensor[i].reshape(1, max_len), \
                segments_tensors[i].reshape(1, max_len), input_masks_tensors[i].reshape(1, max_len))
            q_v = output[0].reshape(-1, 768).to(self.device)
            vectors = torch.cat((vectors, q_v), 0)
        vectors = vectors[1:]
        # output [sentence_num, 768]
        return self.lm_model.work(vectors)


    def forward(self, tokens_tensor, segments_tensors, input_masks_tensors):
        #将输入的句子[sentence_num, max_len]，提取对应的表示, 拼接到一起 [sentence_num = 40, 768]
        vectors = torch.randn(1, 768).to(self.device)
        sentence_num, max_len = tokens_tensor.size()
        for i in range(sentence_num):
            output = self.bert_model(tokens_tensor[i].reshape(1, max_len), \
                segments_tensors[i].reshape(1, max_len), input_masks_tensors[i].reshape(1, max_len))
            q_v = output[0].reshape(-1, 768).to(self.device)
            vectors = torch.cat((vectors, q_v), 0)
        vectors = vectors[1:]
        # print(vectors.size())

        # 根据得到的结果，产生正负样本
        inp, correct, future_hard, previous_hard, future_easy, previous_easy = self.sample_generation(vectors)

        # 使用prefix模型做预测
        output, loss, acc1, acc2 = self.lm_model(inp, correct, future_hard, previous_hard, future_easy, previous_easy)
        return loss, acc1, acc2
