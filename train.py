import torch
import os
from data import DataLoader
from model import PrefixPredict
from optim import Optim
from adam import AdamWeightDecayOptimizer

import warnings
warnings.filterwarnings("ignore")

data_path = os.path.join(os.getcwd(), 'data')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_every = 100

# model construct
model = PrefixPredict(device)
model = model.double()
model = model.to(device)

# optimizer construct
optimizer = Optim(model_size=768, factor=1, warmup=10000,\
    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

# load data
train_data = DataLoader(data_path, 10)


batch_acm = 0
acc_acm, loss_acm =0., 0.
stoa_acc = 0.
# input data (sentence_num, input_dim) = (10, 768)
while True:
    model.train()
    for input_data, correct_data, future_data, previous_data in train_data:
        batch_acm += 1

        # move data too GPU
        input_data = input_data.to(device)
        correct_data = correct_data.to(device)
        future_data = future_data.to(device)
        previous_data = previous_data.to(device)
        
        model.zero_grad()
        res, loss, acc = model(input_data, correct_data, future_data, previous_data)
        loss_acm += loss.item()
        acc_acm += acc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch_acm%print_every == 0:
            print('batch_acm %d, loss %.3f, acc %.3f'\
                %(batch_acm, loss_acm/print_every, acc_acm/print_every), flush=True)
            if acc_acm/print_every > stoa_acc:
                stoa_acc = acc_acm/print_every
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_batch_%dacc_%.3f'%(data_path, train_data.epoch_id, batch_acm, stoa_acc))
            loss_acm, acc_acm = 0., 0.


