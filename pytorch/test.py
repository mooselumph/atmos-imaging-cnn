# -*- coding: utf-8 -*-

import argparse
import os, time
# import PIL.Image as Image
import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imsave
from utils import print_log
import data as dg
from forward_models.forward_models import get_forward_model

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
# Test
parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
parser.add_argument('--set_names', default=['Set12'], help='directory of test dataset')
parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
parser.add_argument('--save_result', default=0, type=int, help='save the reconstructed image, 1 or 0')
# Model
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--model_name', default='model.pth', type=str, help='the model name (e.g. model_001.pth)')
# Degradation
parser.add_argument('--forward_model', default='noise', type=str, help='choose the type of degradation')
parser.add_argument('--forward_params', default=list(), nargs='*', help='parameters for degradation model')
args = parser.parse_args()


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def main():

    assert(torch.cuda.is_available())
    
    for set_count,set_cur in enumerate(args.set_names):

        # Select Dataset
        forward_model = get_forward_model(args.forward_model)(*args.forward_params)

        if set_count == 0:
            # Load Model (if first iteration)
            print('===> Loading model')  
            model_dir = os.path.join('models', args.model+'_' + forward_model.tostring())
            assert(os.path.exists(os.path.join(model_dir, args.model_name)))
            model = torch.load(os.path.join(model_dir, args.model_name))

            # Create save directory
            if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
            
            save_dir = os.path.join(args.result_dir, args.model+'_' + forward_model.tostring())
            if not os.path.exists(save_dir): os.mkdir(save_dir)

        # Create Set directory
        set_dir = os.path.join(save_dir, set_cur)
        if not os.path.exists(set_dir): os.mkdir(set_dir)

        # Load Data
        print('===> Loading data')  
        files,xs = dg.load_data(os.path.join(args.set_dir, set_cur))

        # Evaluate Model
        print('===> Evaluating model') 

        psnrs = []
        ssims = []

        model.eval()  # evaluation mode
        model = model.cuda()

        for i, x in enumerate(xs):

                np.random.seed(seed=0)  # for reproducibility

                y = forward_model.apply(x).astype('float32')

                torch.cuda.synchronize()
                start_time = time.time()
                
                y_ = torch.from_numpy(y[np.newaxis,np.newaxis]).cuda()
                x_ = model(y_)  # inference
                x_ = x_.cpu().detach().numpy().squeeze().astype(np.float32)
                
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, files[i], elapsed_time))

                psnr = compare_psnr(x, x_)
                ssim = compare_ssim(x, x_)

                if args.save_result:
                    name, ext = os.path.splitext(os.path.basename(files[i]))
                    show(np.hstack((y, x_)))  # show the image
                    save_result(x_, path=os.path.join(set_dir, name+'_restored'+ext))  # save the denoised image

                psnrs.append(psnr)
                ssims.append(ssim)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(set_dir, 'results.txt'))

        print_log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))

    
if __name__ == '__main__':
    main()




#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())