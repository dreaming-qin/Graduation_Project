import os
import numpy as np
import json
import h5py


r'''通过trad_result文件夹中的size文件获得相应的kbps结果,要运行该函数,需要先运行以下py文件
audio: 需要先运行make_trad_audio.py获得size文件
video: 需要先运行make_trad_video_thread.py获得size文件
text: 需要先运行make_trad_text.py获得size文件
'''

def get_kbps(size_dict,config):
    r'''返回kbps
    '''
    tst_and_val_id_list=get_val_and_tst_utt_id(config)
    time_h5 = h5py.File(os.path.join(config['target_root'],'time.h5'), 'r')
    bits=0
    sec=0
    for id in tst_and_val_id_list:
        sec+=time_h5[id][()]
        bits+=size_dict[id]*8
    kbps=bits/(1000*sec)
    return kbps

def get_val_and_tst_utt_id(config):
    all_utt_ids=[]
    for cvNo in range(1,11):
        val_int2name, _ = get_trn_val_tst(config['target_root'], cvNo, 'val')
        tst_int2name, _ = get_trn_val_tst(config['target_root'], cvNo, 'tst')
        val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
        tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
        all_utt_ids.extend(val_int2name + tst_int2name)
    return all_utt_ids

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def get_video_kbps(config):
    dir_name='./trad_result/V'
    new_lines='qp,kbps\n'
    for file in sorted(os.listdir(dir_name)):
        if file.endswith('size.txt'):
            with open(os.path.join(dir_name,file),'r') as f:
                lines=f.readlines()[1:]
            size_dict={}
            for p in lines:
                p=p.strip().split(',')
                size_dict[p[0]]=int(p[1])
            kbps=get_kbps(size_dict,config)
            s_i=file.find('qp')+2
            e_i=file.find('_size')
            qp=file[s_i:e_i]
            new_lines+='{},{}\n'.format(qp,kbps)
    with open(os.path.join(dir_name,'kbps.txt'),'w') as f:
        f.writelines(new_lines)

def get_audio_kbps(config):
    dir_name='./trad_result/A'
    new_lines='set_kbps,kbps\n'
    for file in sorted(os.listdir(dir_name)):
        if file.endswith('size.txt'):
            with open(os.path.join(dir_name,file),'r') as f:
                lines=f.readlines()[1:]
            size_dict={}
            for p in lines:
                p=p.strip().split(',')
                size_dict[p[0]]=int(p[1])
            kbps=get_kbps(size_dict,config)
            s_i=file.find('kbps')+4
            e_i=file.find('_size')
            set_kbps=file[s_i:e_i]
            new_lines+='{},{}\n'.format(set_kbps,kbps)
    with open(os.path.join(dir_name,'kbps.txt'),'w') as f:
        f.writelines(new_lines)

def get_text_kbps(config):
    dir_name='./trad_result/L'
    new_lines='kbps\n'
    for file in sorted(os.listdir(dir_name)):
        if file.endswith('size.txt'):
            with open(os.path.join(dir_name,file),'r') as f:
                lines=f.readlines()[1:]
            size_dict={}
            for p in lines:
                p=p.strip().split(',')
                size_dict[p[0]]=int(p[1])
            kbps=get_kbps(size_dict,config)
            new_lines+='{}\n'.format(kbps)
    with open(os.path.join(dir_name,'kbps.txt'),'w') as f:
        f.writelines(new_lines)

if __name__=='__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    # config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config_path = './data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))

    get_video_kbps(config)
    get_audio_kbps(config)
    get_text_kbps(config)