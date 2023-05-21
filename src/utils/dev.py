import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import d_type, last_model, best_model, check_dir
from network.net_model import NetModel
from network.net_loss import NetLoss


def proj_param():
    parser = argparse.ArgumentParser(description='msQSM: Morphology-based Self-supervised Deep Learning for '
                                                 'Quantitative Susceptibility Mapping')
    parser.add_argument('-d', '--device', default='0', type=str)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-g', '--gamma', default=0.01, type=float, choices=[0, 0.01, 0.05, 0.1, 0.5])
    parser.add_argument('-s', '--save_interval', type=int, default=5)
    parser.add_argument('-v', '--validation_interval', type=int, default=5)
    return parser.parse_args()


def set_env(device):
    torch.set_default_dtype(torch.float32)
    if device != 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        return torch.device('cuda')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return torch.device('cpu')


def get_data_loader(args, train_set, valid_set):
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.batch_size // 2
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=False,
        batch_size=min(4, args.batch_size // 2)
    )

    return train_loader, valid_loader


def save_model(epoch, model, opt, best_valid, val_loss, grad, interval):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'best_valid': min(val_loss, best_valid)
    }
    torch.save(state, last_model(grad))
    if val_loss < best_valid:
        torch.save(state, best_model(grad))

    if not epoch or (epoch+1) % interval == 0:
        torch.save(state, f'{check_dir}/{grad}/model_{epoch+1:03d}.pkt')


def load_model(device, model_file):
    torch.set_default_dtype(d_type)
    epoch, best_valid = 0, 999

    loss_func = NetLoss()
    model = NetModel().to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    if os.path.exists(model_file):
        state = torch.load(model_file, map_location=device)
        epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer_state'])
        best_valid = state['best_valid'] if 'best_valid' in state else best_valid

        print(f'  {model_file} loaded. epoch: {epoch}')

    return epoch, model, loss_func, opt, best_valid
