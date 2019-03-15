# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data as dg
from data import GenericDataset
from utils import print_log
from profiles import get_profile


def main():
    assert(torch.cuda.is_available())

    # Get profile
    p = get_profile()

    # Get forward model
    forward_model = p.get_forward_model()

    # Load Model
    print('===> Loading model')    
    model,model_dir,initial_epoch = p.get_model()

    # Loss Function Selection
    criterion = p.loss()

    # Optimizer Selection & Setup
    optimizer = optim.Adam(model.parameters(), lr=p.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    # Cuda Setup
    model = model.cuda()
    # device_ids = [0]
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # criterion = criterion.cuda()

    # Load Data
    print('===> Loading data')  
    xs = dg.gen_data(p)
    dataset = GenericDataset(xs,forward_model)

    # Training
    model.train()

    print('===> Training')
    for epoch in range(initial_epoch, p.epoch):

        scheduler.step(epoch)  # step to the learning rate in this epoch

        # Setup Data Loader
        DLoader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=p.batch_size, shuffle=True)

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
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.shape[0]//p.batch_size, loss.item()/p.batch_size))

        elapsed_time = time.time() - start_time

        # Log and Save
        print_log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(model_dir, 'model_%03d.pth' % (epoch+1)))


if __name__ == '__main__':
    main()