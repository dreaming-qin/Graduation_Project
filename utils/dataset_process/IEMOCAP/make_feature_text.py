import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


import sys
sys.path.append('/home/haojun/docker/code/Graduation_Project/Graduation_Project_baseline')

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

def read_file(file_name):
    lines=''
    with open(file_name,'r') as f:
        lines=f.readlines()
    ans_dict={}
    for line in lines:
        line=line.strip().split(' ')
        utt_id=line[0]
        text=line[2:]
        ans_dict[utt_id]=text
    return ans_dict

def make_all_bert(config):
    # from debug import show_wdseg, show_sentence
    extractor = BertExtractor(cuda=True, cuda_num=1)
    word_info_dir = os.path.join(config['data_root'], 'Session{}/dialog/transcriptions/{}.txt')
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feat_feature_root'], 'raw', "L", "raw_bert.h5")
    h5f = h5py.File(feat_save_path, 'w')
    listed_set=set()
    all_set=set(all_utt_ids)
    for utt_id in tqdm(all_utt_ids):
        # if utt_id!='Ses01F_impro01_F005':
        #     continue
        if utt_id in listed_set:
            continue
        session_id = int(utt_id[4])
        file_name=utt_id[:utt_id.rfind('_')]
        word_info_path = word_info_dir.format(session_id, file_name)
        text_map=read_file(word_info_path)
        id_list=[]
        feat_list=[]
        for key2,value2 in text_map.items():
            if key2 in all_set:
                listed_set.add(key2)
                token_ids, _ = extractor.tokenize(value2)
                utt_feats, _ = extractor.get_embd(token_ids)
                utt_feats = utt_feats.squeeze(0).cpu()
                id_list.append(key2)
                feat_list.append(utt_feats)
        # feat_list=pad_sequence(feat_list, batch_first=True, padding_value=0)
        for key2,value2 in zip(id_list,feat_list):
            utt_group = h5f.create_group(key2)
            utt_group['feat'] = value2.cpu().numpy()
    h5f.close()

if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 创建文件夹
    save_dir = os.path.join(config['feat_feature_root'], 'raw')
    for modality in ['A', 'V', 'L']:
        modality_dir = os.path.join(save_dir, modality)
        if not os.path.exists(modality_dir):
            os.makedirs(modality_dir)

    # make raw feat record with timestamp
    make_all_bert(config)

