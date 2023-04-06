import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import shutil
import subprocess


cmd = 'opusenc --cvbr --bitrate 2 ../dataset/IEMOCAP_full_release/Session2/sentences/wav/Ses02F_script03_1/Ses02F_script03_1_M001.wav ./tmp/input.opus'
lines=subprocess.getstatusoutput(cmd)[1]
lines=lines.split('\n')
for line in lines:
    line=line.strip()
    if line.startswith('Bitrate'):
        s_i=line.find('Bitrate')+9
        e_i=line.find('kbit')
        kbps=float(line[s_i:e_i])
print(kbps)
a=1

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
        cmd = '{}/build/progsrc/smilextract/SMILExtract -C {}/config/compare16/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'.format(
            self.opensmile_tool_dir,self.opensmile_tool_dir, wav, save_path)
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

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

def normlize_on_trn(config,input_file,output_file):
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

def encode_and_decode(config):
    r'''在这里实现encode和decode
    '''

    def encode(wav_file):
        r'''wav_file是输入文件
        wav文件-> opus文件
        返回执行command后的输出, 用于后续获得result'''
        # 对第一步, 由于face中的命名问题需要创建一个新的文件夹来复制图片并重命名图片使其符合ffmpeg要求
        # 命名格式是: 0.1000.png -> 0001.png
        tmp_dir='./tmp'
        makedirs(tmp_dir)

        cmd = "opusenc --bitrate {} {} {}".format(config['kbps'],wav_file, os.path.join(tmp_dir,'input.opus')) 
        lines=subprocess.getstatusoutput(cmd)[1]
        return lines
        
    def get_result(utt_id,lines,save_file):
        r'''从lines中获得结果并以append的形式追加到save_file中
        '''
        lines=lines.split('\n')
        for line in lines:
            line=line.strip()
            if line.startswith('Overhead'):
                s_i=line.find('Overhead')+9
                e_i=line.find('%')
                overhead=float(line[s_i:e_i])
            if line.startswith('Wrote'):
                s_i=line.find('Wrote')+6
                e_i=line.find('bytes')
                byte=int(line[s_i:e_i])
        size=round(byte*(1-overhead/100))
        
        makedirs(os.path.dirname(save_file))
        with open(save_file,'a') as f:
            f.writelines('{},{}\n'.format(utt_id,size))
        return

    def decode(save_file):
        makedirs(os.path.dirname(save_file))
        tmp_dir='./tmp'
        command = 'opusdec {} {}'.format(os.path.join(tmp_dir,'input.opus'),save_file)
        os.system(command)
        shutil.rmtree(tmp_dir)
        return
    
    save_file='./trad_result/A/kbps{}_size.txt'.format(config['kbps'])
    with open(save_file,'w') as f:
        f.writelines('name,size(Byte)\n')
    all_utt_ids = get_all_utt_id(config)
    print('start encoding and decoding')
    for utt_id in all_utt_ids:
        session = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        wav_file=os.path.join(config['data_root'],'Session{}/sentences/wav/{}/{}.wav'.format(session,dialog_id,utt_id))
        lines=encode(wav_file)
        save_file='./trad_result/A/kbps{}_size.txt'.format(config['kbps'])
        get_result(utt_id,lines,save_file)
        save_file=os.path.join(config['trad_data_root'],'A/kbps{}/{}.wav'.format(config['kbps'],utt_id))
        decode(save_file)

def get_feature(config):
    print('start getting feature')
    extractor = OpenSMILEExtractor(opensmile_tool_dir='/home/haojun/docker/opensmile-3.0.1',
        tmp_dir=os.path.join(config['feature_root'],'openSMILEfeature'),
        no_tmp=True)
    all_utt_ids = get_all_utt_id(config)
    all_h5f = h5py.File(os.path.join(config['trad_feature_root'],'A/kbps{}.h5'.format(config['kbps'])), 'w')
    for utt_id in tqdm(all_utt_ids):
        wav_path = os.path.join(config['trad_data_root'],'A/kbps{}/{}.wav'.format(config['kbps'],utt_id))
        feat = extractor(wav_path)
        all_h5f[utt_id] = feat
    all_h5f.close()
    normlize_on_trn(config,os.path.join(config['trad_feature_root'],'A/kbps{}.h5'.format(config['kbps'])),
                    os.path.join(config['trad_feature_root'],'A/kbps{}_mean_std.h5'.format(config['kbps'])))

def make_trad_audio(config):
    r'''供外部py文件的函数调用'''
    encode_and_decode(config)
    get_feature(config)

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
    kbps_list=[2,3,5,7,9]
    for kbps in kbps_list:
        config['kbps']=kbps
        make_trad_audio(config)