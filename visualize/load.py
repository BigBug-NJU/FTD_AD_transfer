'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import nibabel as nib
import numpy as np
from scipy import ndimage

def drop_invalid_range(volume):
    """
    Cut off the invalid area
    """
    zero_value = volume[0, 0, 0]
    non_zeros_idx = np.where(volume != zero_value)
    
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    
    return volume[min_z:max_z, min_h:max_h, min_w:max_w]

def random_center_crop(input_D, input_H, input_W, data):
    """
    Random crop
    """
    index = np.random.randint(0,8)
    crop_data = np.zeros([input_D-8, input_H-8, input_W-8], dtype=np.float32)
    crop_data[:, :, :] = data[index:(index+input_D-8), index:(index+input_H-8), index:(index+input_W-8)]
    return crop_data

def head_crop(data):
    from random import random
    """
    Random crop
    """
    [depth, height, width] = data.shape
    data1 = data[40:(depth-16),0:(height-8),:]
    #print(data1.shape)
    [d1, h1, w1] = data1.shape
    index = np.random.randint(0,8)
    crop_data = np.zeros([d1-8, h1-8, w1], dtype=np.float32)
    crop_data[:, :, :] = data1[index:(index+d1-8), index:(index+h1-8), :]
    return crop_data

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def resize_data(input_D, input_H, input_W, data):
    """
    Resize the data to the input size
    """ 
    [depth, height, width] = data.shape
    scale = [input_D*1.0/depth, input_H*1.0/height, input_W*1.0/width]  
    data = ndimage.interpolation.zoom(data, scale, order=0)

    return data

def crop_data(input_D, input_H, input_W, data):
    """
    Random crop with different methods:
    """ 
    # random center crop
    data = random_center_crop(input_D, input_H, input_W, data)
    return data

def nii_loader(path, input_D, input_H, input_W):
    
    # inp = image.load_img(path, dtype=("float32"))
    # pdb.set_trace()
    nii = nib.load(path)
    data0 = nii.get_data()
    #print(type(data0))
    #print(data0.shape)
    #print(data0.dtype)
    org_shape = data0.shape
    #
    data1 = data0.squeeze()
    # transpose the data from WHD format to DHW
    data2 = np.transpose(data1, (2,1,0))

    # drop out the invalid range
    #data = drop_invalid_range(data)
    
    # resize data
    # data3 = resize_data(input_D, input_H, input_W, data2)

    # crop data
    #data4 = crop_data(input_D, input_H, input_W, data3) 
    # data4 = head_crop(data3)

    # normalization datas
    #data5 = itensity_normalize_one_volume(data4)
    data5 = itensity_normalize_one_volume(data2)

    data6 = data5.astype("float32")
    [z, y, x] = data6.shape
    data = np.reshape(data6, [1, 1, z, y, x])
    return data
