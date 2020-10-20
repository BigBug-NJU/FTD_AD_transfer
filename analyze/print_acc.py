'''
Written by Jingjing Hu, shawkin@yeah.com
'''

import argparse
import os
parser = argparse.ArgumentParser(description='sort files by name')
parser.add_argument('--org', type=str, help='fpath to be sorted')
args = parser.parse_args()

key1 = " * Acc@1 "
key2 = "][  0"
key3 = "][ 0"
acclist = []
loslist = []
with open(args.org, 'r') as f1:
    for line in f1:
        if line.find(key1) == 0:
            tmp = line[9:16]
            acclist.append(tmp)
        if line.find(key2) > 0 or line.find(key3) > 0:
            #print(line)
            ind = line.find("Loss ")
            tmp1 = line[(ind+5):(ind+11)]
            tmp2 = line[(ind+13):(ind+15)]
            if line[ind+12] == "+":
                num = float(tmp1)*pow(10,int(tmp2))
            else:
                num = float(tmp1)*pow(10,-1*int(tmp2))
            loslist.append(str(num))
(filepath,tempfilename) = os.path.split(args.org)
newname = filepath+"/acc_"+tempfilename
with open(newname, 'w') as f2:
    for item in acclist:
        f2.writelines(item)
        f2.writelines('\n')
    f2.close()
print("save: "+newname)
newname = filepath+"/loss_"+tempfilename
with open(newname, 'w') as f3:
    for item in loslist:
        f3.writelines(item)
        f3.writelines('\n')
    f3.close()
print("save: "+newname)
