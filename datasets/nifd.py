'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

from vision import VisionDataset
import os.path

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, sets, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.input_D = sets.input_D+8
        self.input_H = sets.input_H+8
        self.input_W = sets.input_W+8
        self.phase = sets.phase

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path, self.input_D, self.input_H, self.input_W)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.nii')

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
    from random import random
    """
    Random crop
    """
    index = np.random.randint(0,8)
    crop_data = np.zeros([input_D-8, input_H-8, input_W-8], dtype=np.float32)
    crop_data[:, :, :] = data[index:(index+input_D-8), index:(index+input_H-8), index:(index+input_W-8)]
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
    import nibabel as nib
    import numpy as np
    # inp = image.load_img(path, dtype=("float32"))
    # pdb.set_trace()
    nii = nib.load(path)
    data1 = nii.get_data().squeeze()
    # transpose the data from WHD format to DHW
    data2 = np.transpose(data1, (2,1,0))

    # drop out the invalid range
    #data = drop_invalid_range(data)
    
    # resize data
    data3 = resize_data(input_D, input_H, input_W, data2)

    # crop data
    data4 = crop_data(input_D, input_H, input_W, data3) 

    # normalization datas
    data5 = itensity_normalize_one_volume(data4)

    data6 = data5.astype("float32")
    [z, y, x] = data6.shape
    data = np.reshape(data6, [1, z, y, x])
    return data

class NiiFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/ad/xxx.nii
        root/ad/xxy.nii
        root/ad/xxz.nii

        root/nifd/123.nii
        root/nifd/nsdf3.nii
        root/nifd/asd932_.nii

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, sets=None, transform=None, target_transform=None,
                 loader=nii_loader, is_valid_file=None):
        super(NiiFolder, self).__init__(root, sets, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
