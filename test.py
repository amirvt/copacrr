__author__ = 'vandermonde'

import os
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
import shutil


input_dir = "/media/amir/6B254F8510DE287D/Amir/data/cosine/old"
output_dir = "/media/amir/6B254F8510DE287D/Amir/data/cosine/old_bin"



def convertAndWriteFile(in_f, outf):
    obj = genfromtxt(in_f, delimiter=",").astype(np.float32)
    if len(obj.shape) == 1:
        obj = np.expand_dims(obj, axis=0)[:,:-1]
    else:
        obj = obj[:,:-1]

    np.save(outf,obj)

input_feat_dirs = [ o for o in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,o))]
for ifd in input_feat_dirs:

    input_query_dirs = [o for o in os.listdir(input_dir+"/"+ifd) if os.path.isdir(os.path.join(input_dir+"/"+ifd,o))]
    for iqd in input_query_dirs:
        if not os.path.exists(output_dir+"/"+ifd+"/"+iqd):
            os.makedirs(output_dir+"/"+ifd+"/"+iqd)

        file_names = [o for o in os.listdir(input_dir+"/"+ifd+"/"+iqd)]
        print("feature:" + ifd + "\tquery:" + iqd + "\tfiles: " + str(len(file_names)))
        for fn in tqdm(file_names):
            convertAndWriteFile(input_dir+"/"+ifd+"/"+iqd+"/"+fn, output_dir+"/"+ifd+"/"+iqd+"/"+fn)

        shutil.rmtree(input_dir+"/"+ifd+"/"+iqd)


