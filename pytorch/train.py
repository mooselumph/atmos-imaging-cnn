# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data as dg
from data import GenericDataset
from models import get_model
from forward_models.forward_models import get_forward_model
from utils import print_log


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
# Model
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
# Degradation
parser.add_argument('--forward_model', default='noise', type=str, help='choose the type of degradation')
parser.add_argument('--forward_params', default=list(), nargs='*', help='parameters for degradation model')
# Training
parser.add_argument('--train_data', default='data/Train/Train400', type=str, help='path of train data')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch
     
def main():
    assert(torch.cuda.is_available())
    batch_size = args.batch_size
    n_epoch = args.epoch

    # Select Dataset
    forward_model = get_forward_model(args.forward_model)(*args.forward_params)

    # Setup Save Dir
    save_dir = os.path.join('models', args.model+'_' + forward_model.tostring())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Model Selection
    print('===> Loading model')    
    # Load Model from Previous Point
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    else:
        model = get_model(args.model)
        model = model()

    # Loss Function Selection
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()

    # Optimizer Selection & Setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    # Cuda Setup
    model = model.cuda()
    # device_ids = [0]
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # criterion = criterion.cuda()

    # Load Data
    print('===> Loading data')  
    xs = dg.gen_data(data_dir=args.train_data)
    dataset = GenericDataset(xs,forward_model)

    # Training
    model.train()

    print('===> Training')
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epoch

        # Setup Data Loader
        DLoader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)

        epoch_loss = 0

        # Process Batches
        start_time = time.time()
        for n_count, batch_yx in enumerate(DLoader):

                optimizer.zero_grad()
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()

                loss = criterion(model(batch_y), batch_x) # Calculate Loss
                epoch_loss += loss.item()

                loss.backward() # Calculate gradients
                optimizer.step() # Update parameters

                if n_count % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.shape[0]//batch_size, loss.item()/batch_size))

        elapsed_time = time.time() - start_time

        # Log and Save
        print_log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))


if __name__ == '__main__':
    main()