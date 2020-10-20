'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import os
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch split AD filenames as imagenet')
parser.add_argument('list', metavar='DIR',
                    help='txt file')
parser.add_argument('--per', default=0.8, type=float,
                    help='percentage of train dateset(default=0.8)')

def split_files(filelist, train_per):
    tensor_filenames = [line.rstrip() for line in open(filelist, 'r')]
    N = len(tensor_filenames)
    assert (N <= 10000)
    np.random.shuffle(tensor_filenames)
    train_name = "./train_" + filelist
    train_file = open(train_name, 'a')
    val_name = "./val_" + filelist
    val_file = open(val_name, 'a')
    for i in range(int(N * train_per)):
        train_file.write(tensor_filenames[i] + '\n')
    for j in range(int(N * train_per), N):
        val_file.write(tensor_filenames[j] + '\n')
    train_file.close()
    val_file.close()

def main():
    args = parser.parse_args()
    org_files = Path(args.list)
    if not org_files.exists():
        print("Cann't locate " + args.list + ", check!")
        return

    save_names = "./train+" + args.list
    save_files = Path(save_names)
    if save_files.exists():
        os.remove(save_names)

    save_names = "./test_" + args.list
    save_files = Path(save_names)
    if save_files.exists():
        os.remove(save_names)

    split_files(args.list, args.per)

if __name__ == '__main__':
    main()
