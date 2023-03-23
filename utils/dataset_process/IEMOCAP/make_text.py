import os
import h5py
import json
import numpy as np
from tqdm import tqdm

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

def calc_real_time2(frm):
    frm = int(frm)
    return frm / 100

def read_align_fil2(file):
    lines = open(file).readlines()[1:-1]
    ans = []
    for line in lines:
        line = line.strip()
        sfrm, efem, _, word = line.split()
        st = calc_real_time(sfrm)
        et = calc_real_time(efem)
        if word.startswith('<') and word.endswith('>'):
            continue
        if word.startswith('++'):
            word.replace('+', '')

        if '(' in word:
            word = word[:word.find('(')].lower()
        else:
            word = word.lower()
        
        if len(word) == 0:
            continue
        one_record = {
            'start_time':st,
            'end_time':et,
            'word': word,
        }
        ans.append(one_record)
    
    return ans

def make_all_bert2(config):
    # from debug import show_wdseg, show_sentence
    extractor = BertExtractor(cuda=True, cuda_num=0)
    word_info_dir = os.path.join(config['data_root'], 'Session{}/sentences/ForcedAlignment/{}')
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feature_root'], 'raw', "L", "raw_bert.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(all_utt_ids):
        # print("UTT_ID:", utt_id)
        if utt_id == 'Ses03M_impro03_M001': # 这个语句缺少对齐信息文件
            continue
        if utt_id=='Ses01F_impro01_F005':
            a=1
        session_id = int(utt_id[4])
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        word_info_path = os.path.join(word_info_dir.format(session_id, dialog_id), utt_id + '.wdseg')
        word_infos = read_align_file(word_info_path)
        word_lst = [x["word"] for x in word_infos]
        # print("WORDS:", word_lst)
        token_ids, word_idxs = extractor.tokenize(word_lst)
        utt_start = [word_infos[i]['start_time'] for i in word_idxs]
        utt_end = [word_infos[i]['end_time'] for i in word_idxs]
        utt_feats, _ = extractor.get_embd(token_ids)
        utt_feats = utt_feats.squeeze(0).cpu().numpy()[1:-1, :]
        assert utt_feats.shape[0] == len(utt_end)
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = utt_feats
        utt_group['start'] = utt_start
        utt_group['end'] = utt_end
        # show_wdseg(utt_id)
        # show_sentence(utt_id)
        # input()
    h5f.close()

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
    extractor = BertExtractor(cuda=True, cuda_num=0)
    word_info_dir = os.path.join(config['data_root'], 'Session{}/dialog/transcriptions/{}.txt')
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feature_root'], 'raw', "L", "raw_bert.h5")
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
        for key2,value2 in text_map.items():
            if key2 in all_set:
                listed_set.add(key2)
                token_ids, _ = extractor.tokenize(value2)
                utt_feats, _ = extractor.get_embd(token_ids)
                utt_feats = utt_feats.squeeze(0).cpu().numpy()
                utt_group = h5f.create_group(key2)
                utt_group['feat'] = utt_feats
    h5f.close()

if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 创建文件夹
    save_dir = os.path.join(config['feature_root'], 'raw')
    for modality in ['A', 'V', 'L']:
        modality_dir = os.path.join(save_dir, modality)
        if not os.path.exists(modality_dir):
            os.makedirs(modality_dir)

    # make raw feat record with timestamp
    make_all_bert(config)

