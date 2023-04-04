import shutil
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 
import json
import h5py
from tqdm import tqdm
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)


from utils.dataset_process.tools.densenet import DenseNet

r'''获得视频传统编解码的数据库, 编解码信息, 对应的raw feature特征
首先需要运行make_feature_video.py获得需要的脸部图片, 需要完成以下步骤
1. 先获得传统编解码后的数据集, 在这过程中获得需要传输的数据大小
2.抓取rawfeature特征'''

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
time_step=0.1

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_all_utt_id(config):
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    return all_utt_ids

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def encode_and_decode(config):
    r'''在这里实现encode和decode'''
    def encode(png_dir,VVCdir):
        r'''png_dir是裁剪好的脸部png图片
        png图片序列-> yuv
        1.将png序列变成yuv拖入至VVCdir中
        2.使用命令行运行EncoderAppStaticd'''
        # 对第一步, 由于face中的命名问题需要创建一个新的文件夹来复制图片并重命名图片使其符合ffmpeg要求
        # 命名格式是: 0.1000.png -> 0001.png
        tmp_dir='./tmp'
        for png_file in os.listdir(png_dir):
            cnt=int(float(png_file.split('.png')[0])*10)
            new_name='{:.4d}.png'.format(cnt)
            new_png_file=os.path.join(tmp_dir, new_name)
            old_png_file=os.path.join(png_dir, png_file)
            shutil.copyfile(old_png_file,new_png_file)
        # 将图片变成yuv格式
        cmd='cd {} &&ffmpeg -y -r 10 -i %4d.png -pix_fmt yuv420p -s 64x64 input.yuv'.format(tmp_dir)
        os.system(cmd)
        shutil.copyfile(os.path.join(tmp_dir,'input.yuv',os.path.join(VVCdir,'input.yuv')))
        shutil.rmtree(tmp_dir)

        frames=len(os.listdir(png_dir))
        # 第二步:使用命令行运行EncoderAppStaticd
        cmd = "cd {} && ./EncoderAppStaticd  -c encoder_lowdelay_vtm.cfg -q {} -f {} > Enc_Out.txt".format(
                VVCdir, config['qp'], frames)
        print(cmd)
        os.system(cmd)
        return 
        
    def get_result(utt_id,VVCdir,save_file):
        r'''在编码后会获得二进制流文件str.bin, 获得结果并以append的形式追加到save_csv_file中'''
        makedirs(os.path.dirname(save_file))
        bin_file=os.path.join(VVCdir,'str.bin')
        byte=os.path.getsize(bin_file)
        with open(save_file,'a') as f:
            f.writelines('{},{}\n'.format(utt_id,byte))
        return

    def decode(VVCdir,save_png_dir):
        r'''save_png_dir是解码后的png图片序列
        yuv->png图片序列
        1. 将bin解码为yuv
        2. 将yuv变为png图片序列'''
        # 第一步: 将bin解码为yuv
        command = "cd {} && ./DecoderAppStaticd -b str.bin -o output.yuv > Dec_Out.txt".format(
                VVCdir)
        print(command)
        os.system(command)
        shutil.copyfile(os.path.join(VVCdir,'output.yuv',os.path.join(save_png_dir,'output.yuv')))
        # 第二步: 将yuv变为png图片序列
        command = "cd {} && ffmpeg -s 64x64 -i output.yuv %4d.png".format(
                save_png_dir)
        print(command)
        os.system(command)
        return
    
    all_utt_ids = get_all_utt_id(config)
    print('start encoding and decoding')
    for utt_id in tqdm(all_utt_ids):
        session = utt_id[4]
        png_dir=os.path.join(config['data_root'],'Session{}/face/{}'.format(session,utt_id))
        VVCdir='./VVCSoftware_VTM-VTM-15.0/encode_video_demo'
        encode(png_dir,VVCdir)
        save_file='./trad_result/V/qp{}.txt'.format(config['qp'])
        get_result(utt_id,VVCdir,save_file)
        save_png_dir=os.path.join(config['trad_data_root'],'V/qp{}/{}'.format(config['qp'],utt_id))
        decode(VVCdir,save_png_dir)

def get_feature(config):
    def extract_feature(model,img_paths):
        normMean = [0.49139968, 0.48215827, 0.44653124]
        normStd = [0.24703233, 0.24348505, 0.26158768]
        normTransform = transforms.Normalize(normMean, normStd)
        video_transform= transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            normTransform
        ])
        clip=[]
        model=model.to(device)
        for img_path in img_paths:
            img=Image.open(img_path).convert('RGB')
            clip.append(video_transform(img))
        clip = torch.stack(clip, 0).to(device)
        model.eval()
        with torch.no_grad():
            out=model(clip)
        return out 

    extractor =DenseNet(growthRate=12, depth=100, reduction=0.5,bottleneck=True, nClasses=10,pre_train=True,device=device)
    all_utt_ids = get_all_utt_id(config)
    data_root=os.path.join(config['trad_data_root'],'V/qp{}'.format(config['qp']))
    feat_save_path = os.path.join(config['trad_feature_root'], "V/qp{}.h5".format(config['qp']))
    h5f = h5py.File(feat_save_path, 'w')
    print('start getting feature')
    for utt_id in all_utt_ids:
        png_dir=os.path.join(data_root,utt_id)
        utt_face_pics = sorted(glob.glob(os.path.join(png_dir, '*.png')), key=lambda x: float(os.path.basename(x).split('.png')[0]))
        utt_feats = extract_feature(extractor,utt_face_pics) 
        utt_feats=[a.cpu().numpy() for a in utt_feats]
        if len(utt_feats) != 0:
            utt_feats = np.array(utt_feats)
        else:
            print('missing face')
            utt_feats = np.zeros([1, 342])
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定模态遗失
        if 'Ses05F_script02_' in utt_id and utt_id[-4]=='M':
            print('{} missing video'.format(utt_id))
            utt_feats=np.zeros(utt_feats.shape)
        h5f[utt_id]=utt_feats
    h5f.close()

    pass

def make_all_face(config):
    r'''供外部py文件的函数调用'''
    encode_and_decode(config)
    get_feature(config)

if __name__ == '__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    # 添加传统编解码qp信息
    config['qp']=56
    # 创建文件夹
    save_dir_list = [os.path.join(config['trad_feature_root'], 'raw'),
        config['trad_data_root'],'trad_result']
    for save_dir in save_dir_list:
        for modality in ['A', 'V', 'L']:
            modality_dir = os.path.join(save_dir, modality)
            if not os.path.exists(modality_dir):
                os.makedirs(modality_dir)

    make_all_face()