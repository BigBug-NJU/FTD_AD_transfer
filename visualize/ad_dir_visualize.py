'''
Calculate contribution graphs of all nii images in one directory
Written by Jingjing Hu, shawkin@yeah.com
'''

import torch
from torch.nn import ReLU
import os
import argparse

from ad_utils import (get_ad_params, save_gradient_images_dir)

class GuidedBackprop3D():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        #self.hook_hjj()

    def hook_hjj(self):
        def hook_fn(module, grad_in, grad_out):
            print("bn1_layer bw:")
            for go in grad_out:
                print(go.shape)
            print("<--------------------")
            for gi in grad_in:
                print(gi.shape)
        def hook_fw_function(module, ten_in, ten_out):
            print("bn1_layer fw:")
            for ti in ten_in:
                print(ti.shape)
            print("-->")
            for to in ten_out:
                print(to.shape)
        bn1_layer = list(self.model.module._modules.items())[1][1]
        #print(bn1_layer)
        bn1_layer.register_backward_hook(hook_fn)
        bn1_layer.register_forward_hook(hook_fw_function)

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # print("first_layer bw:")
            # for go in grad_out:
            #     print(go.shape)
            # print("<--------------------")
            # print(type(grad_in), len(grad_in))
            # for index in range(len(grad_in)):
            #     if grad_in[index] is None:
            #         print(index, grad_in[index])
            #     else:
            #         print(index, grad_in[index].shape)
            # for gi in grad_in:
            #     print(gi.shape)
            self.gradients = grad_in[0]
        def hook_fw_function(module, ten_in, ten_out):
            print("first_layer fw:")
            for ti in ten_in:
                print(ti.shape)
            print("-->")
            for to in ten_out:
                print(to.shape)
        # Register hook to the first layer
        # first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.model.module._modules.items())[0][1]
        #first_layer = self.model.module.conv1
        #print(first_layer)
        first_layer.register_backward_hook(hook_function)
        #first_layer.register_forward_hook(hook_fw_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            # print("relu_bw:")
            # for gi in grad_in:
            #     print(gi.shape)
            # print("-->")
            # for go in grad_out:
            #     print(go.shape)
            # print("fo:", self.forward_relu_outputs[-1].shape, "mo:", modified_grad_out.shape)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            # print("relu_fw:")
            # for ti in ten_in:
            #     print(ti.shape)
            # print("-->")
            # for to in ten_out:
            #     print(to.shape)
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        ##
        modules = []
        modules.append(self.model.module.relu)
        for pos, bottle in self.model.module.layer1._modules.items():
            for ind, module in bottle._modules.items():
                if isinstance(module, ReLU):
                    #print(pos, ind, module)
                    modules.append(module)
        for pos, bottle in self.model.module.layer2._modules.items():
            for ind, module in bottle._modules.items():
                if isinstance(module, ReLU):
                    #print(pos, ind, module)
                    modules.append(module)
        for pos, bottle in self.model.module.layer3._modules.items():
            for ind, module in bottle._modules.items():
                if isinstance(module, ReLU):
                    #print(pos, ind, module)
                    modules.append(module)
        for pos, bottle in self.model.module.layer4._modules.items():
            for ind, module in bottle._modules.items():
                if isinstance(module, ReLU):
                    #print(pos, ind, module)
                    modules.append(module)
        for module in modules:
            #print(module)
            module.register_backward_hook(relu_backward_hook_function)
            module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        if not args.no_cuda:
            one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        if not args.no_cuda:
            gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        else:
            gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir',
        default='/data/hjj/scp_AD_DCM/ConvertedNII',
        type=str,
        help='Path for image need test')
    parser.add_argument(
        '--save_dir',
        default='/data/hjj/scp_AD_DCM/ConvertedNII_cg',
        type=str,
        help='Path to save')
    parser.add_argument(
        '--target',
        default=0,
        type=int,
        help="Number of segmentation classes")
    parser.add_argument(
        '--input_D',
    	default=256,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=240,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=160,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes")
    parser.add_argument(
        '--num_classes',
        default=2,
        type=int,
        help="Number of classification classes")
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    args_org = parser.parse_args()
    
    args = args_org 
    lsdir = os.listdir(args_org.img_dir)
    files = [i for i in lsdir if os.path.isfile(os.path.join(args_org.img_dir, i))]
    for f in files:
        args.img_path = os.path.join(args_org.img_dir, f)
        (prep_img, file_name_to_export, pretrained_model) = get_ad_params(args)
        if not args.no_cuda:
            prep_img = prep_img.cuda()
        ## very important !!!
        prep_img.requires_grad_(True)
        #print("input", prep_img.shape)
        #print("     :", prep_img.requires_grad)
        # Guided backprop
        GBP = GuidedBackprop3D(pretrained_model)
        # Get gradients cpu
        guided_grads = GBP.generate_gradients(prep_img, args.target)
        #print(guided_grads.shape)
        # Save colored gradients
        save_gradient_images_dir(guided_grads, args.save_dir, file_name_to_export + '_Guided_BP_color')
    print('Guided backprop completed')
# docker attach hjj_main
# python ad_dir_visualize.py --img_dir=/data/hjj/scp_AD_DCM/ConvertedNII --save_dir=/data/hjj/scp_AD_DCM/ConvertedNII_cg --target=0 --resume_path=./save_3/org_model_best.pth.tar --num_classes=3 --gpu_id 4
