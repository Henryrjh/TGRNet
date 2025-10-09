from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import stereo_datasets as datasets
from TGRNet.TGRNet import create_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def sequence_loss(disp_preds, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of disp predictions """
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = F.smooth_l1_loss(disp_gt[valid.bool()],disp_preds[i][valid.bool()], reduction='mean')
        disp_loss += i_weight * i_loss            

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, total_steps=0):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(args.path, 'log'))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.path, 'log'))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.path, 'log'))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args):

    model = create_model()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model_weight = checkpoint['model']
        model.load_state_dict(model_weight, strict=True)
        logging.info(f"Done loading checkpoint")   

        total_steps = checkpoint['iters']
        logging.info(f"Resume from iters {total_steps}...")

        logger = Logger(model, scheduler, total_steps)     

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        total_steps = 0

    model.cuda()
    model.train()
    model.freeze_bn()

    save_freq = args.save_freq

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = total_steps
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()
            image1, image2, disp, valid = [x.cuda() for x in data_blob]

            assert model.training            
            disp_predictions = model(image1, image2, iters=args.train_iters)
 
            assert model.training

            loss, metrics = sequence_loss(disp_predictions, disp, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if (total_steps % save_freq == save_freq - 1):
                save_path = Path(os.path.join(args.path, 'ckpt') + '/%d_%s.pth' % (total_steps + 1, args.model_name))
                save_data = {'iters': global_batch_num,
                             'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                             }

                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(save_data, save_path)
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(os.path.join(args.path, 'ckpt') + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.model_name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = os.path.join(args.path, 'ckpt') + '/%s.pth' % args.model_name
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")

    parser.add_argument('--model_name', default='TGRNet',help="restore checkpoint")
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--path', default='./checkpoints/whu')

    ### traing
    parser.add_argument('--left_dir', default='/home/rjh/Stereo_matching/dataset/WHU-Stereo_dataset/with_GT/train/left', help="")
    parser.add_argument('--right_dir', default='/home/rjh/Stereo_matching/dataset/WHU-Stereo_dataset/with_GT/train/right', help="")
    parser.add_argument('--disp_dir', default='/home/rjh/Stereo_matching/dataset/WHU-Stereo_dataset/with_GT/train/disp', help="")
    parser.add_argument('--training', default=True, help="")
    parser.add_argument('--max_disp', default=64, help="")
    parser.add_argument('--min_disp', default=-128, help="")
    parser.add_argument('--gt_size', default=256, help="")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--save_freq', type=int, default=5000, help="batch size used during training.")

    # Training parameters
    parser.add_argument('--train_datasets', nargs='+', default=['rs'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")

    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')
 
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,  
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(os.path.join(args.path, 'ckpt')).mkdir(exist_ok=True, parents=True)
    train(args)