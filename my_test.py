

import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import shutil


def get_val_and_tst_utt_id(config):
    config['target_root']='../dataset/IEMOCAP_features_2021/target'
    all_utt_ids=[]
    for cvNo in range(1,11):
        val_int2name, _ = get_trn_val_tst(config['target_root'], cvNo, 'val')
        tst_int2name, _ = get_trn_val_tst(config['target_root'], cvNo, 'tst')
        val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
        tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
        all_utt_ids.append(val_int2name + tst_int2name)
    return all_utt_ids

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


if __name__ == '__main__':
    # in_data2021 = h5py.File('/home/haojun/docker/code/Graduation_Project/dataset/IEMOCAP_features_2021/V/denseface.h5', 'r')
    # in_data_test = h5py.File('/home/haojun/docker/code/Graduation_Project/dataset/IEMOCAP_features_test/raw/V/efficientface.h5', 'r')
    # feat_size=50
    # for key2 in in_data2021.keys():
    #     data2021=in_data2021[key2][()]
    #     data_test=in_data_test[key2][()]
    #     if data_test.shape[0]!=feat_size:
    #         print('-2')
    #     if data2021.shape[0]!=feat_size:
    #         print('-1')
    # load config
    config_path = './data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))

    # make raw feat record with timestamp
    get_val_and_tst_utt_id(config)
        
