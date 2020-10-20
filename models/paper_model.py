'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import torch
from torch import nn
from models import resnet
import os

def generate_res50(opt):
    model = resnet.resnet50(
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes)
        
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.conv_seg = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    model.fc = nn.Linear(2048, opt.num_classes)
    if not opt.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    return model

def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.conv_seg = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    model.fc = nn.Linear(2048, opt.num_classes)
    return h2d_model(model, opt)

def h2d_model(model, opt):
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            import os
            gpu_ids = ""
            for gpu_id in opt.gpu_id:
                gpu_ids += str(gpu_id)
                gpu_ids += ","
            gpu_ids = gpu_ids[0:-1]
            os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

            opt.gpu_id = range(len(opt.gpu_id))
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            #model = nn.DataParallel(model, device_ids=opt.gpu_id).cuda()
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
