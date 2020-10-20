'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch List AD Training nii files')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--key', type=str,
                    help='key to be found')

def print_files(father_path, key_name, save_names):
    save_file = open(save_names, 'a')
    lsdir = os.listdir(father_path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(father_path, i))]
    if dirs:
        for i in dirs:
            if (i.find(key_name) >= 0):
                print_files(os.path.join(father_path, i), key_name, save_names)

    files = [i for i in lsdir if os.path.isfile(os.path.join(father_path, i))]

    for f in files:
        if os.path.splitext(f)[1] == ".nii":
            sss = (os.path.join(father_path, f)) + '\n'
            # print(sss)
            save_file.write(sss)

    save_file.close()
    return

def main():
    args = parser.parse_args()
    save_names = "./" + args.key + ".txt"
    save_files = Path(save_names)
    if save_files.exists():
        os.remove(save_names)
    print_files(args.data, args.key, save_names)

if __name__ == '__main__':
    main()

# python list_name.py /data/NIFD_Patient --key=NIFD