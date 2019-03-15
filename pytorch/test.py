# -*- coding: utf-8 -*-

import os, time
import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imsave
from utils import print_log
import data as dg
from models import DnCNN
from profiles import get_profile

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

def main():

    p = get_profile()

    assert(torch.cuda.is_available())
    
    for set_count,set_cur in enumerate(p.test_sets):

        # Select Dataset
        forward_model = p.get_forward_model()

        if set_count == 0:
            # Load Model (if first iteration)
            print('===> Loading model')  
            model = p.get_test_model()

            # Create save directory
            result_dir = p.get_result_dir()

        # Create Set directory
        set_dir = os.path.join(result_dir, set_cur)
        if not os.path.exists(set_dir): os.mkdir(set_dir)

        # Load Data
        print('===> Loading data')  
        files,xs = dg.load_data(os.path.join(p.test_dir, set_cur))

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

                if p.save_result:
                    name, ext = os.path.splitext(os.path.basename(files[i]))
                    save_result(np.hstack((x, y, x_)), path=os.path.join(set_dir, name+'_restored'+ext))  # save the denoised image

                psnrs.append(psnr)
                ssims.append(ssim)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if p.save_result:
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