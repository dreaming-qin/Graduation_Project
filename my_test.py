

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
    in_data_test = h5py.File('/home/haojun/docker/code/Graduation_Project/dataset/IEMOCAP_features_test/raw/V/efficientface.h5', 'r')
    feat_size=50
    for key2 in in_data2021.keys():
        data2021=in_data2021[key2][()]
        data_test=in_data_test[key2][()]
        if data_test.shape[0]!=feat_size:
            print('-2')
        if data2021.shape[0]!=feat_size:
            print('-1')
        
