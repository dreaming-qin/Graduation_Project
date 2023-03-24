import os
import h5py
import json
import numpy as np
import shutil
from tqdm import tqdm

# import sys
# sys.path.append('/home/haojun/docker/code/Graduation_Project/Graduation_Project_baseline')


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

def make_aligned_data(config):
    raw_A_path = os.path.join(config['feature_root'], 'aligned', "A", "raw_comparE.h5")
    raw_V_path = os.path.join(config['feature_root'], 'aligned', "V", "raw_efficientface.h5")
    raw_L_path = os.path.join(config['feature_root'], 'aligned', "L", "raw_bert.h5")
    raw_A = h5py.File(raw_A_path, 'r')
    raw_V = h5py.File(raw_V_path, 'r')
    raw_L = h5py.File(raw_L_path, 'r')
    all_utt_ids = get_all_utt_id(config)
    aligned_A_path = os.path.join(config['feature_root'], 'aligned', "A", "aligned_comparE.h5")
    aligned_V_path = os.path.join(config['feature_root'], 'aligned', "V", "aligned_efficientface.h5")
    aligned_L_path = os.path.join(config['feature_root'], 'aligned', "L", "aligned_bert.h5")
    aligned_A_h5f = h5py.File(aligned_A_path, 'w')
    aligned_V_h5f = h5py.File(aligned_V_path, 'w')
    aligned_L_h5f = h5py.File(aligned_L_path, 'w')

    for utt_id in tqdm(all_utt_ids):
        if utt_id == 'Ses03M_impro03_M001': # 这个语句缺少对齐信息文件
            continue
        utt_A_feat, utt_A_start, utt_A_end = \
            raw_A[utt_id]['feat'][()], raw_A[utt_id]['start'][()], raw_A[utt_id]['end'][()]
        utt_V_feat, utt_V_start, utt_V_end = \
            raw_V[utt_id]['feat'][()], raw_V[utt_id]['start'][()], raw_V[utt_id]['end'][()]
        utt_L_feat, utt_L_start, utt_L_end = \
            raw_L[utt_id]['feat'][()], raw_L[utt_id]['start'][()], raw_L[utt_id]['end'][()]
        
        utt_aligned_A, utt_aligned_V = [], []
        for word_start, word_end in zip(utt_L_start, utt_L_end):
            word_aligned_a = calc_word_aligned(word_start, word_end, utt_A_feat, utt_A_start, utt_A_end, default_dim=130)
            word_aligned_v = calc_word_aligned(word_start, word_end, utt_V_feat, utt_V_start, utt_V_end, default_dim=1024)
            utt_aligned_A.append(word_aligned_a)
            utt_aligned_V.append(word_aligned_v)
        
        utt_aligned_A = np.array(utt_aligned_A)

        utt_aligned_V = np.array(utt_aligned_V)
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定模态遗失
        if 'Ses05F_script02_' in utt_id and utt_id[-4]=='M':
            utt_aligned_V = np.zeros(utt_aligned_V.shape)
        assert(len(utt_aligned_A) == len(utt_aligned_V) == len(utt_L_feat))
        # print(f'A:{utt_aligned_A.shape} V:{utt_aligned_V.shape} L:{utt_L_feat.shape}')
        aligned_A_h5f[utt_id] = utt_aligned_A
        aligned_V_h5f[utt_id] = utt_aligned_V
        aligned_L_h5f[utt_id] = utt_L_feat

def calc_word_aligned(word_start, word_end, frame_feats, frame_start, frame_end, default_dim=342):
    _frame_set = []
    assert word_end > word_start
    for feat, start, end in zip(frame_feats, frame_start, frame_end):
        if start == end == -1 and np.sum(frame_feats) == 0:
            break
        assert end > start, f'{start}, {end}, {frame_feats}'
        if start > word_end or end < word_start:
            continue
        else:
            _frame_set.append(feat)
    if len(_frame_set) > 0:
        _frame_set = np.array(_frame_set)
    else:
        _frame_set = np.zeros([1, default_dim])
    return np.mean(_frame_set, axis=0)

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

if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 创建文件夹
    save_dir = os.path.join(config['feature_root'], 'aligned')
    for modality in ['A', 'V', 'L']:
        modality_dir = os.path.join(save_dir, modality)
        if not os.path.exists(modality_dir):
            os.makedirs(modality_dir)

    # make_aligned_data
    # make_aligned_data(config)

    # # normalize A feat
    # normlize_on_trn(config,
    #     os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE.h5'), 
    #     os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE_mean_std.h5')
    # )

    # 复制文件
    
    shutil.copyfile(os.path.join(config['feature_root'], 'aligned', "A", "aligned_comparE.h5"),
        os.path.join(config['feature_root'], "A", 'comparE.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'aligned', "A", "aligned_comparE_mean_std.h5"),
        os.path.join(config['feature_root'], "A", 'comparE_mean_std.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'aligned', "V", "aligned_efficientface.h5"),
        os.path.join(config['feature_root'], "V", 'efficientface.h5'))
    shutil.copyfile(os.path.join(config['feature_root'], 'aligned', "L", "aligned_bert.h5"),
        os.path.join(config['feature_root'], "L", 'bert_large.h5'))

