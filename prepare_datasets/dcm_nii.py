'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import os
import argparse
from pathlib import Path
#import pdb
parser = argparse.ArgumentParser(description='Convert downloaded dcm to nii')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dst', type=str,
                    help='file_path to save as')

def convert_files(father_path, newpath):
    lsdir = os.listdir(father_path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(father_path, i))]
    if dirs:
        for i in dirs:
            convert_files(os.path.join(father_path, i), newpath)
    
    files = [i for i in lsdir if os.path.isfile(os.path.join(father_path, i))]
    for f in files:
       if (f.find(".dcm") >= 0):
           #os.system("cd " + father_path)
           os.chdir(father_path)
           os.system("rm -f ./*.nii.gz")
           os.system("/hjj/dcm2nii *")
           ind = f.find("_br_raw_20")
           #pdb.set_trace()
           newname = "t_" + f[0:(ind+16)] + ".nii.gz"
           filepath = os.path.join(newpath, newname)
           os.system("mv 20*.nii.gz " + filepath)
           return
    return

def main():
    args = parser.parse_args()
    newpath = args.dst
    os.system("rm -rf " + newpath)
    os.system("mkdir -p " + newpath)
    convert_files(args.data, newpath)

if __name__ == '__main__':
    main()

# python dcm_nii.py /data/NIFD_P_3T_T1_MPRAGE/NIFD/ --dst=/data/NIFD_ConvertedNII
