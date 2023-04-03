import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import heapq
from collections import Counter
import shutil


import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)

r'''获得传统方法的文本kbps, 并获得相应特征
除了数据集IEMOCAP, 需要有视频持续时间time.h5, 进行特征压缩时已经提取出的raw feature文件bert-large.h5 (即需要运行make_feature_text.py文件s)
对数据集处理后会生成适用于传统方法的数据集在IEMOCAP_features_trad中, 对应特征在IEMOCAP_features_trad中,
传统编码kbps结果在trad_result中
'''


# 哈夫曼编码的实现
def huffman_encode(text):
    # 统计字符出现的次数
    freq = Counter(text)
    
    # 使用优先队列存储频率和字符
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    
    # 构建哈夫曼树
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # 构建编码表
    huff_dict = dict(heapq.heappop(heap)[1:])
    
    # 对文本进行编码
    encoded_text = "".join([huff_dict[char] for char in text])
    
    # 返回编码后的文本和编码表
    return encoded_text, huff_dict

def get_huffman_size(text):
        # 对文本进行哈夫曼编码
    encoded_text, _ = huffman_encode(text)
    # 将编码后的内容写入新文件
    with open("compressed_file.bin", "wb") as f:
        f.write(bytearray([int(encoded_text[i:i+8], 2) for i in range(0, len(encoded_text), 8)]))
    # 获取压缩后的文件的大小
    compressed_size = os.path.getsize("compressed_file.bin")
    os.remove('compressed_file.bin')
    return compressed_size


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
    # 对于特征获得，text是无损压缩，直接搬之前的特征
    if not os.path.exists(os.path.join(config['trad_feature_root'],'L')):
        os.makedirs(os.path.join(config['trad_feature_root'],'L'))
    shutil.copyfile(os.path.join(config['feature_root'],'L/bert_large.h5'),
        os.path.join(config['trad_feature_root'],'L/bert_large.h5'))
    word_info_dir = os.path.join(config['data_root'], 'Session{}/dialog/transcriptions/{}.txt')
    all_utt_ids = get_all_utt_id(config)
    time_h5f=h5py.File(os.path.join(config['target_root'], 'time.h5'), 'r')
    feat_save_path = os.path.join(config['trad_feature_root'], 'raw', "L", "raw_bert.h5")
    h5f = h5py.File(feat_save_path, 'w')
    listed_set=set()
    all_set=set(all_utt_ids)
    kbps_str='name,kbps\n'
    for utt_id in tqdm(all_utt_ids):
        if utt_id in listed_set:
            continue
        session_id = int(utt_id[4])
        file_name=utt_id[:utt_id.rfind('_')]
        word_info_path = word_info_dir.format(session_id, file_name)
        text_map=read_file(word_info_path)
        for key2,value2 in text_map.items():
            if key2 in all_set:
                listed_set.add(key2)
                # 将切割好的文本放到传统编解码数据集中
                with open(os.path.join(config['trad_data_root'],'L/{}.txt'.format(key2)),'w') as f:
                    f.writelines(' '.join(value2))
                # 计算哈夫曼编码后的大小
                compressed_size = get_huffman_size(' '.join(value2))
                sec=time_h5f[key2][()]
                kbps=(compressed_size*8)/(1000*sec)
                kbps_str+='{},{}\n'.format(key2,kbps)
    h5f.close()
    with open('./trad_result/L/kbps.txt','w') as f:
        f.writelines(kbps_str)


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
    make_all_bert(config)

