# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
from forward_models.isoplanatic import apply_aberration as apply_isoplanatic

# TODO: Make these inputs arguments to gen_data function
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

class GenericDataset(Dataset):
    
    def __init__(self, xs, forward_model):
        super(GenericDataset, self).__init__()
        self.xs = xs
        self.forward_model = forward_model

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = self.forward_model.apply(batch_x.squeeze())
        batch_y = batch_y[np.newaxis].astype(np.float32)  # equivalent to np.expand_dims(batch_y,axis=0)

        return torch.from_numpy(batch_y), torch.from_numpy(batch_x)

    def __len__(self):
        return self.xs.shape[0]


def aug_img(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = aug_img(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def gen_data(data_dir='data/Train/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')

    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch normalization
    data = np.delete(data, range(discard_n), axis=0)

    # TODO: Determine redundancy with the above ^^
    data = data.transpose((0, 3, 1, 2)).astype('float32')/255.0  # tensor of the clean patches, NxCxHxW

    return data


def load_data(data_dir='data/Test/Set68'):
    # Load images for testing
    ims = []
    files = []
    for f in glob.glob(data_dir+'/*.*'):
        if f.endswith(".jpg") or f.endswith(".bmp") or f.endswith(".png"):
            files.append(f)
            im = np.array(cv2.imread(f,0), dtype=np.float32)/255.0
            ims.append(im)

    return files,ims

if __name__ == '__main__': 

    data = gen_data(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       