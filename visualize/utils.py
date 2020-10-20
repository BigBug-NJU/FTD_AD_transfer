'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from load import nii_loader
import nibabel as nib
import sys
sys.path.append("..")
from models.paper_model import generate_res50

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.nii')
    save_image(gradient, path_to_file)

def save_gradient_images_dir(gradient, savename, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists(savename):
        os.makedirs(savename)
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(savename, file_name + '.nii')
    save_image(gradient, path_to_file)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint16)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    #print("save_img dtype:", im.dtype)
    #if isinstance(im, (np.ndarray, np.generic)):
    #    im = format_np_output(im)
    im1 = im.squeeze()
    im = np.transpose(im1, (2,1,0))
    #print(type(im))
    #print(im.shape)
    #print(im.dtype)
    im = im.astype(np.float64)
    new_image = nib.Nifti1Image(im, np.eye(4))
    #print("save_img new dtype:", im.dtype)
    #new_image.set_data_dtype(np.uint16)
    new_image.set_data_dtype(np.float64)
    print(path)
    nib.save(new_image, path)

def get_ad_params(sets):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    file_name_to_export = sets.img_path[sets.img_path.rfind('/')+1:sets.img_path.rfind('.')]
    # Read image
    prep_img = nii_loader(sets.img_path, sets.input_D, sets.input_H, sets.input_W)
    t_img = torch.from_numpy(prep_img)
    # Define model
    model = generate_res50(sets)
    if os.path.isfile(sets.resume_path):
        #print("=> loading checkpoint '{}'".format(sets.resume_path))
        checkpoint = torch.load(sets.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        #print("=> loaded checkpoint '{}' (epoch {})"
        #      .format(sets.resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(sets.resume_path))
    return (t_img,
            file_name_to_export,
            model)
