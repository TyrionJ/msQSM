import os
import torch

d_type = torch.float32

check_dir = f'../checkpoints'
last_model = lambda g: f'{check_dir}/{g}/latest.pkl'
best_model = lambda g: f'{check_dir}/{g}/best.pkl'
record = lambda g: f'{check_dir}/{g}/record.txt'
pretrained_model = f'../checkpoints/msQSM.pkl'

data_dir = f'../data'
train_file = f'{data_dir}/train.hd'
valid_file = f'{data_dir}/valid.hd'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

gms = [0, 0.01, 0.05, 0.1, 0.5]
if not os.path.exists(f'{check_dir}/{gms[0]}'):
    for gamma in gms:
        os.makedirs(f'{check_dir}/{gamma}')
