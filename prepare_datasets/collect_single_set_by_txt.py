'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch split AD filenames as imagenet')
parser.add_argument('list', metavar='DIR',
                    help='txt file, train/val_AD/MCI/NC/NIFD.txt')
parser.add_argument('--dst', type=str,
                    help='dst dir to save datasets')

def main():
    args = parser.parse_args()
    org_files = Path(args.list)
    if not org_files.exists():
        print("Cann't locate " + args.list + ", check!")
        return

    type = args.list[0:args.list.find('_')]
    kind = args.list[(args.list.find('_')+1) : args.list.find('.txt')]
    newpath = os.path.join(args.dst, type)
    newpath = os.path.join(newpath, kind)
    os.system("rm -rf " + newpath)
    os.system("mkdir -p " + newpath)

    tensor_filenames = [line.rstrip() for line in open(args.list, 'r')]
    N = len(tensor_filenames)

    for k in range(N):
        if (tensor_filenames[k].find(kind) >= 0):
            cmd = "cp " + tensor_filenames[k] + " " + newpath
            #print(cmd)
            os.system(cmd)
        else:
            print("not found " + kind +"!!! Check " + tensor_filenames[k])

if __name__ == '__main__':
    main()
