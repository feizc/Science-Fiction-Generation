import numpy as np 
import torch 
import os
import json


def batchify(inp, correct, previous, future):
    return torch.from_numpy(inp).double(), torch.from_numpy(correct).double(),\
        torch.from_numpy(previous).double(), torch.from_numpy(future).double()

class DataLoader(object):

    def __init__(self, filepath, batch_size):
        super(DataLoader, self).__init__()
        self.batch_size = batch_size
        self.input_data = np.loadtxt(os.path.join(filepath, 'input.txt'))
        self.correct_data = np.loadtxt(os.path.join(filepath, 'correct.txt'))
        self.previous_data = np.loadtxt(os.path.join(filepath, 'previous.txt'))
        self.future_data = np.loadtxt(os.path.join(filepath, 'future.txt'))
        self.epoch_id = 0

    def __iter__(self):
        idx = 0
        self.epoch_id += 1
        while idx + self.batch_size < len(self.input_data):
            yield batchify(self.input_data[idx:idx+self.batch_size], self.correct_data[idx:idx+self.batch_size],\
                self.previous_data[idx:idx+self.batch_size], self.future_data[idx:idx+self.batch_size])
            idx += self.batch_size

class RawDataLoader(object):
    """读取句子"""
    def __init__(self, file_path, sentence_num = 40):
        super(RawDataLoader, self).__init__()
        self.data = []
        files = os.listdir(file_path)
        for file in files:
            file_name = os.path.join(file_path, file)
            with open(file_name, 'r', encoding='utf8') as load_f:
                load_dict = json.load(load_f)
                t_sentences = load_dict['content']
                self.data = self.data + t_sentences

        self.epoch_id = 0
        self.sentence_num = sentence_num

    def __iter__(self):
        idx = 0
        self.epoch_id +=1
        while idx + self.sentence_num < len(self.data):
            yield self.data[idx:idx+self.sentence_num]
            idx += self.sentence_num
