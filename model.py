import torch
from torch import nn
import torch.nn.functional as F 

from utils import gelu, LayerNorm
from transformer import TransformerLayer, SelfAttentionMask

class ContrativeLoss(nn.Module):
    def __init__(self, device, factor=1.0):
        super(ContrativeLoss, self).__init__()
        self.device = device
        self.factor = factor

    def accuracy_compute(self, positive_pro, negative_pro1, negative_pro2):
        assert len(positive_pro)==len(negative_pro1)==len(negative_pro2)
        correct_num = 0
        for i in range(len(positive_pro)):
            if positive_pro[i] > negative_pro1[i] and positive_pro[i] > negative_pro2[i]:
                correct_num += 1
        return correct_num / len(positive_pro)

    def forward(self, output, correct, future, previous):
        # output = [sentence_num, 768]
        output_norm = torch.norm(output, dim=1)
        positive_pro = torch.sum(output*correct, dim=1) / torch.norm(correct, dim=1) / output_norm
        negative_pro1 = torch.sum(output*future, dim=1) / torch.norm(future, dim=1) / output_norm
        negative_pro2 = torch.sum(output*previous, dim=1) / torch.norm(previous, dim=1) / output_norm
        # probability = [sentence_num, ]
        loss = - torch.log(torch.exp(positive_pro) / (torch.exp(positive_pro)\
            + self.factor * torch.exp(negative_pro1) + self.factor * torch.exp(negative_pro2)))
        accuracy = self.accuracy_compute(positive_pro, negative_pro1, negative_pro2)
        return loss, accuracy


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

    def forward(self, inp, correct, future, previous):
        seq_len, input_dim = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        # x = [sentence_num, bsz, input_dim]
        x = inp.unsqueeze(1)
        for layer in self.layers:
            x, _, _ = layer(x, self_attn_mask = self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        x = x.squeeze(1)
        # loss = [sentence_num, ]
        loss, acc = self.loss_fun(x, correct, future, previous)

        return x, loss.mean(), acc

