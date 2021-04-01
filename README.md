
# Idea, code and pretrained model
This repository contains a Pytorch implementation of [https://doi.org/10.3389/fnins.2020.626154]. 

Precise identification of frontotemporal dementia (FTD) and Alzheimer's disease (AD), which are the two most common types of dementia in the younger-elderly population, is advantageous for targeted treatment and patient care. The atrophy of specific brain regions shown by structural magnetic resonance imaging (MRI) is an important part of the diagnostic criteria for FTD and AD. However, MRI-based diagnosis mainly relies on the professional knowledge and clinical experience of doctors.

We investigated the diagnostic value of deep learning (DL)-based networks in differentiating between patients with FTD, patients with AD and normal control (NC), on an individual patient basis. To the best of our knowledge, this is the first successful trial of the diagnosis of FTD with an end-to-end DL technology. The proposed approach achieved promising performance without any professional intervention. The pattern knowledge learned is generalizable, and can be transferred to other datasets and tasks. That is, DL will free researchers from endless problems. Furthermore, we applied a gradient visualization algorithm based on guided backpropagation to calculate the contribution graph. It told us intuitively that the proposed network mined the potential patterns that may be different from human clinicians, which may provide new insight into the understanding of FTD and AD.

### Contents
1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Demo](#Demo)
4. [Code](#Code)
5. [Data](#Data)
6. [Dataset](#Dataset)
7. [Train](#Train)
8. [Visualize](#Visualize)
9. [Analyze](#Analyze)
10. [Acknowledgement](#Acknowledgement)

### Requirements
- docker 18.06.1
- Python 3.6.9
- PyTorch 1.2.0
- CUDA Version 10.0

### Installation
1. pull docker image
```
docker pull pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
```
2. create docker container
```
docker run --runtime=nvidia -it --name FTD_main -v /home/hjj/:/hjj pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
```
3. install pytorch
```
pip install -r requirements.txt
```

### Demo
```
python transfer_paper.py --gpu_id 0 1 2 3 4 5 6 7 --data_root=/data/ResHjjTrain/ --pretrain_path=./pretrained/resnet_50_23dataset.pth --batch_size=12 --n_epochs=1000 --num_classes=3 --num_workers=1 --input_D=256 --input_H=240 --input_W=160 > hjj_paper_e6_192224160_0.001.log
```


### Code
```
git clone https://github.com/BigBug1992/AD_NIFD_transfer.git
cd AD_NIFD_Med3D_transfer
```

### Data
collect volumes from:
https://ida.loni.usc.edu/login.jsp?project=NIFD

uncompress them and junk paths 
```
cd prepare_datasets
unzip -d ./NIFD_P_3T_T1_MPRAGE -j /downloads/NIFD_P_3T_T1_mprage_xx.zip
```

### Dataset
```python
# convert dcm to nii
python dcm_nii.py /data/NIFD_Patient_3T_T1_MPRAGE/NIFD/ --dst=/data/NIFD_Patient

# print nii names, get NIFD.txt
python list_name.py /data/NIFD_Patient --key=NIFD

# sort names, get sorted_NIFD.txt
# Considering that the same subject may be scanned multiple times at multiple time points, once the test data participates in the training process in any form, it will cause data leakage and result in unreasonable model evaluation. 
python sort_name.py --org=NIFD.txt

# split names, get train_sorted_NIFD.txt test_sorted_NIFD.txt 
# The loose dataset is divided into training set and testing set randomly at subject level according to the ratio of 4:1.
python split_list.py sorted_NIFD.txt --per=0.8

# adjust the txt above to make sure the volumes are divided at subject level
# get train_NIFD.txt test_NIFD.txt

# collect dataset by txt
# get:
# /data/scenario3
#             |--train/
#             |   |--NIFD/: a lot of imgaes named xxx.nii
#             |--test/
#             |   |--NIFD/
python collect_single_set_by_txt.py train_NIFD.txt --dst=/data/scenario3

# Repeat the above actions until you have your own dataset, like: 
# /data/MyDataset
#     |--train/
#     |   |--Class1/: a lot of imgaes named xxx.nii
#     |   |--Class2/
#     |   |--Class3/ ...
#     |--test/
#     |   |--Class1/
#     |   |--Class2/
#     |   |--Class3/ ...
```

### Train

train main scenario and get: 
hjj_paper_e6_192224160_0.001.log  
checkpoint.pth.tar 
model_best.pth.tar 
"record_" + cur_time + ".txt"

```
python transfer_paper.py --gpu_id 0 1 2 3 4 5 6 7 --data_root=/data/ResHjjTrain/ --pretrain_path=./pretrain/resnet_50_23dataset.pth --batch_size=12 --n_epochs=1000 --num_classes=3 --num_workers=1 --input_D=256 --input_H=240 --input_W=160 > hjj_paper_e6_192224160_0.001.log
```

train sub scenario and get: 
hjj_paper_e5_2_192224160_0.001.log 
roc_checkpoint.pth.tar
roc_model_best.pth.tar 
"roc_record_" + cur + ".txt"

```
python transfer_sub.py --gpu_id 0 1 2 3 4 5 6 7 --data_root=/data/ROC/ResHjjTrain/ --pretrain_path=./save_e5/checkpoint.pth. --batch_size=12 --n_epochs=1000 --num_classes=2 --num_workers=1 --input_D=256 --input_H=240 --input_W=160 > hjj_paper_e5_2_192224160_0.001.log
```

Med3D pre-trained models can be found at ([Google Drive](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing) or [Tencent Weiyun](https://share.weiyun.com/55sZyIx))

Pre-trained models and logs for 9 scenarios in paper [Deep learning-based classification and voxel-based visualization of frontotemporal dementia and Alzheimer’s disease] can be found at ([BaiduNetdisk](https://pan.baidu.com/s/1ro-3jFbxUKHYhK5jp4uFaQ), passcode is "fc0q").

### Visualize
cd visualize

calculate single nii image:
```
python visualize.py --img_path=/hjj/map_test_5/20111130_150955t1mprages022a1001.nii --target=2 --resume_path=./save_e6/checkpoint.pth.tar --num_classes=3 --gpu_id 4
```

calculate all nii images in one directory:
```
python dir_visualize.py --img_dir=/data/NIFD_Patient --save_dir=/data/NIFD_Patient_cg --target=0 --resume_path=./save_e6/checkpoint.pth.tar --num_classes=3 --gpu_id 0 1 2 3 4 5 6 7
```

### Analyze
cd analyze
```
python print_acc.py --org=../save_e4/hjj_paper_base_Mprage_e4_192224160_0.001.log
```

### Acknowledgement
We thank [Med3D-net](https://github.com/Tencent/MedicalNet) and [MRBrainS18](https://mrbrains18.isi.uu.nl/) for their releasing code.
