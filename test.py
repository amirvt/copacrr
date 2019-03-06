__author__ = 'vandermonde'

import os
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
import shutil


input_dir = "/media/vandermonde/My Passport/runs/run6/data"
output_dir = "/media/vandermonde/32b27fef-6be0-4b5c-b235-71e906ef2818/runs"



def convertAndWriteFile(in_f, outf):
    obj = genfromtxt(in_f, delimiter=",").astype(np.float32)
    if len(obj.shape) == 1:
        obj = np.expand_dims(obj, axis=0)[:,:-1]
    else:
        obj = obj[:,:-1]

    np.save(outf,obj)

# input_feat_dirs = [ o for o in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,o))]
# for ifd in input_feat_dirs:
#
#     input_query_dirs = [o for o in os.listdir(input_dir+"/"+ifd) if os.path.isdir(os.path.join(input_dir+"/"+ifd,o))]
#     for iqd in input_query_dirs:
#         if not os.path.exists(output_dir+"/"+ifd+"/"+iqd):
#             os.makedirs(output_dir+"/"+ifd+"/"+iqd)
#
#         file_names = [o for o in os.listdir(input_dir+"/"+ifd+"/"+iqd)]
#         print("feature:" + ifd + "\tquery:" + iqd + "\tfiles: " + str(len(file_names)))
#         for fn in tqdm(file_names):
#             convertAndWriteFile(input_dir+"/"+ifd+"/"+iqd+"/"+fn, output_dir+"/"+ifd+"/"+iqd+"/"+fn)
#
#         shutil.rmtree(input_dir+"/"+ifd+"/"+iqd)
#

f1 = "/media/vandermonde/HDD/binary_run/run0/data/dfs/350/"
files = [ o for o in os.listdir(f1) if not os.path.isdir(os.path.join(input_dir,o))]
f2 = "/media/vandermonde/HDD/binary_run2/run2/data/dfs/350/"

for file in files:
    first = np.load(f1 + file)
    second = np.load(f2 + file)
    if not np.array_equal(first, second):
        print("are not equal " )

# print("f1.shape", first.shape)
# print("f2.shape", second.shape)
# print("f1", first)
# print("f2", second)

