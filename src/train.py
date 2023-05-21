import torch
from tqdm import tqdm
import time
import warnings

from config import record, train_file, d_type, valid_file, last_model
from network.net_dataset import NetDataset
from utils import load_model, save_model, proj_param, set_env, get_data_loader

warnings.filterwarnings('ignore')


def integrate_loss(l_model, l_mph, gamma):
    return l_model + gamma * l_mph


def train(epoch, data_loader, model, loss_func, opt, device, epoch_num, gama_g):
    model.train()
    avg_loss, avg_cyc, avg_mph = 0, 0, 0

    with tqdm(total=len(data_loader), desc=f'Training {epoch + 1}/{epoch_num}', unit='it') as pbar:
        for N, item in enumerate(data_loader):
            opt.zero_grad()

            x, fld, msk = item
            del item

            x = x.to(device, dtype=d_type)
            y_hat = model(x)

            fld = fld.to(device, dtype=d_type)
            msk = msk.to(device, dtype=d_type)

            l_cycle, l_mph = loss_func(y_hat, x, fld, msk)
            loss = integrate_loss(l_cycle, l_mph, gama_g)

            avg_loss = (avg_loss * N + loss.item()) / (N + 1)
            avg_cyc = (avg_cyc * N + l_cycle.item()) / (N + 1)
            avg_mph = (avg_mph * N + l_mph.item()) / (N + 1)

            loss.backward()
            opt.step()

            pbar.set_postfix(**{'1.avg': '{0:.6f}'.format(avg_loss),
                                '2.bat': '{0:.6f}'.format(loss.item()),
                                '3.cyc': '{0:.6f}'.format(avg_cyc),
                                '4.mph': '{0:.6f}'.format(avg_mph)})
            pbar.update()

    return round(avg_loss, 6)


def validation(data_loader, model, loss_func, device, epoch, gama_g):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f'~~Validation {epoch+1}', unit='it') as pbar:
            for N, item in enumerate(data_loader):
                mix, fld, msk = item
                del item

                mix = mix.to(device, dtype=d_type)
                y_hat = model(mix)

                fld = fld.to(device, dtype=d_type)
                msk = msk.to(device, dtype=d_type)

                l_cycle, l_mph = loss_func(mix, y_hat, fld, msk)
                loss = integrate_loss(l_cycle, l_mph, gama_g)
                avg_loss = (avg_loss * N + loss.item()) / (N + 1)

                pbar.set_postfix(**{'loss': '{0:.6f}'.format(avg_loss)})
                pbar.update()
    return round(avg_loss, 6)


def main(args):
    device = set_env(args.device)

    train_set = NetDataset(train_file)
    valid_set = NetDataset(valid_file)
    train_loader, valid_loader = get_data_loader(args, train_set, valid_set)

    start_epoch, model, loss_func, opt, best_valid = load_model(device, last_model(args.gamma))
    val_loss = best_valid
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)

    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        train_loss = train(epoch, train_loader, model, loss_func, opt, device, args.epoch, args.gamma)
        duration = round(time.time() - start_time, 6)

        if not epoch or (epoch+1) % args.validation_interval == 0:
            val_loss = validation(valid_loader, model, loss_func, device, epoch, args.gamma)
        sched.step()

        with open(record(args.gamma), 'a') as f:
            f.write(f'[{epoch + 1}/{args.epoch}]: {duration}\t'
                    f'{train_loss}\t'
                    f'{val_loss}\n')

        save_model(epoch, model, opt, best_valid, val_loss, args.gamma, args.save_interval)

        if val_loss < best_valid:
            best_valid = val_loss


if __name__ == '__main__':
    main(proj_param())
