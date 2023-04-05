import os
import h5py
import json
import numpy as np
import shutil
from tqdm import tqdm

import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)

from utils.dataset_process.IEMOCAP.make_feature_text import make_all_bert
from utils.dataset_process.IEMOCAP.make_feature_audio import make_all_openSMILE
from utils.dataset_process.IEMOCAP.make_feature_video import make_all_efficientface




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

def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name if utt_id != 'Ses03M_impro03_M001']
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))
    h5f.close()

def format_data(config):
    raw_A_path = os.path.join(config['feature_root'], 'raw', "A", "raw_comparE.h5")
    raw_V_path = os.path.join(config['feature_root'], 'raw', "V", "raw_efficientface.h5")
    raw_L_path = os.path.join(config['feature_root'], 'raw', "L", "raw_bert.h5")
    raw_A = h5py.File(raw_A_path, 'r')
    raw_V = h5py.File(raw_V_path, 'r')
    raw_L = h5py.File(raw_L_path, 'r')
    all_utt_ids = get_all_utt_id(config)
    aligned_A_path = os.path.join(config['feature_root'], 'raw', "A", "comparE.h5")
    aligned_V_path = os.path.join(config['feature_root'], 'raw', "V", "efficientface.h5")
    aligned_L_path = os.path.join(config['feature_root'], 'raw', "L", "bert_large.h5")
    aligned_A_h5f = h5py.File(aligned_A_path, 'w')
    aligned_V_h5f = h5py.File(aligned_V_path, 'w')
    aligned_L_h5f = h5py.File(aligned_L_path, 'w')

    for utt_id in tqdm(all_utt_ids):
        utt_A_feat =  raw_A[utt_id]['feat'][()]
        utt_V_feat= raw_V[utt_id]['feat'][()]
        utt_L_feat= raw_L[utt_id]['feat'][()]
        
        
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定模态遗失
        if 'Ses05F_script02_' in utt_id and utt_id[-4]=='M':
            utt_V_feat = np.zeros(utt_V_feat.shape)
        aligned_A_h5f[utt_id] = utt_A_feat
        aligned_V_h5f[utt_id] = utt_V_feat
        aligned_L_h5f[utt_id] = utt_L_feat

    aligned_A_h5f.close()
    aligned_V_h5f.close()
    aligned_L_h5f.close()

if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 创建文件夹
    save_dir_list = [os.path.join(config['feature_root'], 'raw'),
        config['feature_root']]
    for save_dir in save_dir_list:
        for modality in ['A', 'V', 'L']:
            modality_dir = os.path.join(save_dir, modality)
            if not os.path.exists(modality_dir):
                os.makedirs(modality_dir)

    # # text
    # make_all_bert(config)
    # # audio
    # make_all_openSMILE(config)
    # # video
    # make_all_efficientface(config)

    # format_data
    format_data(config)

    # # normalize A feat
    normlize_on_trn(config,
        os.path.join(config['feature_root'], 'raw', 'A', 'comparE.h5'), 
        os.path.join(config['feature_root'], 'raw', 'A', 'comparE_mean_std.h5')
    )

    # 复制文件
    shutil.copyfile(os.path.join(config['feature_root'], 'raw', "A", "comparE.h5"),
        os.path.join(config['feature_root'], "A", 'comparE.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'raw', "A", "comparE_mean_std.h5"),
        os.path.join(config['feature_root'], "A", 'comparE_mean_std.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'raw', "V", "efficientface.h5"),
        os.path.join(config['feature_root'], "V", 'efficientface.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'raw', "L", "bert_large.h5"),
        os.path.join(config['feature_root'], "L", 'bert_large.h5'))

