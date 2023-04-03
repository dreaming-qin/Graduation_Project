import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import heapq
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)

from utils.dataset_process.tools.bert_extractor import BertExtractor

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def get_all_utt_id(config):
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    return all_utt_ids


def check_name(name):
    if not name.startswith('Ses') :
        return False
    try:
        int(name[-3:])
    except Exception:
        return False
    return True

def read_file(file_name):
    lines=''
    with open(file_name,'r') as f:
        lines=f.readlines()
    ans_dict={}
    for line in lines:
        line=line.strip().split(' ')
        utt_id=line[0]
        if not check_name(utt_id):
            continue
        index=line[1].find('-')
        s_time=float(line[1][1:index])
        e_time=float(line[1][index+1:-2])
        assert e_time>s_time
        ans_dict[utt_id]=e_time-s_time
    return ans_dict

def make_time(config):
    word_info_dir = os.path.join(config['data_root'], 'Session{}/dialog/transcriptions/{}.txt')
    all_utt_ids = get_all_utt_id(config)
    time_h5f=h5py.File(os.path.join(config['target_root'], 'time.h5'), 'w')
    listed_set=set()
    all_set=set(all_utt_ids)
    for utt_id in tqdm(all_utt_ids):
        if utt_id in listed_set:
            continue
        session_id = int(utt_id[4])
        file_name=utt_id[:utt_id.rfind('_')]
        word_info_path = word_info_dir.format(session_id, file_name)
        time_map=read_file(word_info_path)
        id_list=[]
        for key2,value2 in time_map.items():
            if key2 in all_set:
                listed_set.add(key2)
                time_h5f[key2]=value2
                id_list.append(key2)
    time_h5f.close()


if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 创建文件夹
    save_dir_list = [os.path.join(config['trad_feature_root'], 'raw'),
        config['trad_data_root'],'trad_result']
    for save_dir in save_dir_list:
        for modality in ['A', 'V', 'L']:
            modality_dir = os.path.join(save_dir, modality)
            if not os.path.exists(modality_dir):
                os.makedirs(modality_dir)

    # make raw feat record with timestamp
    make_time(config)

