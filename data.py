import numpy as np 
import torch 
import os


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