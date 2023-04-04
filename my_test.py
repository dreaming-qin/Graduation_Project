

import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import shutil


if __name__ == '__main__':
    in_data2021 = h5py.File('/home/haojun/docker/code/Graduation_Project/dataset/IEMOCAP_features_2021/V/denseface.h5', 'r')
    in_data_test = h5py.File('/home/haojun/docker/code/Graduation_Project/dataset/IEMOCAP_features_test/raw/V/raw_efficientface.h5', 'r')
    min_len = 100000
    a=0
    for key2 in in_data2021.keys():
        data2021=in_data2021[key2][()]
        data_test=in_data_test[key2]['feat'][()]
        if data_test.shape[0]==342:
            a+=1
        min_len=min(min_len,data2021.shape[0])
    print(min_len)
    print(a)
