'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import argparse
parser = argparse.ArgumentParser(description='sort files by name')
parser.add_argument('--org', type=str, help='fpath to be sorted')
args = parser.parse_args()

orglist = []
with open(args.org, 'r') as f1:
    for line in f1:
        orglist.append(line.strip())

newname = "sorted_"+args.org
with open(newname, 'w') as f2:
    for item in sorted(orglist):
        f2.writelines(item)
        f2.writelines('\n')
    f2.close()
