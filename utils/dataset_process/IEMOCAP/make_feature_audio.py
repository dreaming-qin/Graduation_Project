import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import shutil


class OpenSMILEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir, downsample=10, tmp_dir='./tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/compare16/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'.format(
            self.opensmile_tool_dir, wav, save_path)
        # print(cmd)
        os.system(cmd)
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = df.iloc[:, 2:]
        if len(wav_data) > self.downsample:
            wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
            if self.no_tmp:
                os.remove(save_path) 
        else:
            wav_data = None
            self.print(f'Error in {wav}, no feature extracted')

        return wav_data


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_all_comparE(config):
    extractor = OpenSMILEExtractor(opensmile_tool_dir='/root/opensmile',
        tmp_dir=os.path.join(config['feature_root'],'openSMILEfeature'),
        no_tmp=True)
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'raw','A', 'comparE.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        wav_path = os.path.join(config['data_root'], f'Session{ses_id}', 'sentences', 'wav', f'{dialog_id}', f'{utt_id}.wav')
        feat = extractor(wav_path)
        all_h5f[utt_id] = feat
    all_h5f.close()

def normlize_on_trn(config):
    input_file=os.path.join(config['feature_root'], 'raw','A', 'comparE.h5')
    output_file=os.path.join(config['feature_root'],'raw', 'A', 'raw_comparE_mean_std.h5')
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
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

def load_A(config):
    feat_path = os.path.join(config['feature_root'], 'raw','A', "raw_comparE.h5")
    feat = h5py.File(feat_path, 'r')
    meta_path = os.path.join(config['feature_root'], 'raw','A', "raw_comparE_mean_std.h5")
    mean_std = h5py.File(meta_path, 'r')
    return feat, mean_std

def make_openSMILE(config):
    feat, _ = load_A(config)
    
    # mean_std file copy
    # mean_std_src = os.path.join(config['feature_root'], 'raw','A', "raw_comparE_mean_std.h5")
    # mean_std_tgt = os.path.join(config['feature_root'], 'raw', "A", "comparE_mean_std.h5")
    # shutil.copyfile(mean_std_src, mean_std_tgt)
    # process feat and record timestamp
    feat_save_path = os.path.join(config['feature_root'], 'raw', "A", "raw_comparE.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(feat.keys()):
        utt_feat = feat[utt_id][()]
        start = np.array([0.1 * x for x in range(len(utt_feat))])
        end = start + 0.1
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = feat[utt_id][()]
        utt_group['start'] = start
        utt_group['end'] = end
    h5f.close()


def make_all_openSMILE(config):
    # make raw feat record with timestamp
    make_all_comparE(config)
    normlize_on_trn(config)
    make_openSMILE(config)


if __name__ == '__main__':
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
    make_all_comparE(config)
    normlize_on_trn(config)
    make_openSMILE(config)