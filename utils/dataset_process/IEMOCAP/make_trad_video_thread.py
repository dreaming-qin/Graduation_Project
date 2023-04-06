import shutil
import os
import numpy as np
import json
import h5py
from multiprocessing import Pool,cpu_count
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)


from utils.dataset_process.tools.densenet import DenseNet

r'''获得视频传统编解码的数据库, 编解码信息, 对应的raw feature特征
首先需要运行make_feature_video.py获得需要的脸部图片, 该函数可以完成以下步骤
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
    def divide_utt_id(id_list,divide_cnt):
        return [id_list[i:i + divide_cnt] for i in range(0, len(id_list), divide_cnt)]
    
    all_utt_ids = get_all_utt_id(config)
    print('start encoding and decoding')

    # 使用多进程方式进行编解码
    thread_cnt=config['thread']
    all_utt_ids=divide_utt_id(all_utt_ids,len(all_utt_ids)//thread_cnt+1)
    p = Pool(min(thread_cnt,cpu_count()))
    for i in range(thread_cnt):
        p.apply_async(_thread_encode_and_decode, args=[config,all_utt_ids[i],i+1])
    print('wait for all threads done...')
    p.close()
    p.join()

    print('thread is finished, start merging result')
    # 整合所有结果
    save_dir='./trad_result/V'
    lines=['name,size(Byte)\n']
    for txt_file in os.listdir(save_dir):
        if txt_file.startswith('qp{}_index'.format(config['qp'])):
            with open(os.path.join(save_dir,txt_file),'r') as f:
                lines2=f.readlines()
            lines.extend(lines2)
    with open('./trad_result/V/qp{}_size.txt'.format(config['qp']),'w') as f:
        f.writelines(lines)
    for txt_file in os.listdir(save_dir):
        if txt_file.startswith('qp{}_index'.format(config['qp'])):
            os.remove(os.path.join(save_dir,txt_file))

def _thread_encode_and_decode(config,utt_id_list,index):
    def encode(png_dir,VVCdir):
        r'''png_dir是裁剪好的脸部png图片
        png图片序列-> yuv
        1.将png序列变成yuv拖入至VVCdir中
        2.使用命令行运行EncoderAppStaticd'''
        # 对第一步, 由于face中的命名问题需要创建一个新的文件夹来复制图片并重命名图片使其符合ffmpeg要求
        # 命名格式是: 0.1000.png -> 0001.png
        tmp_dir='./tmp{}'.format(index)
        makedirs(tmp_dir)
        for png_file in os.listdir(png_dir):
            cnt=int(float(png_file.split('.png')[0])*10)
            new_name='{:0>4d}.png'.format(cnt)
            new_png_file=os.path.join(tmp_dir, new_name)
            old_png_file=os.path.join(png_dir, png_file)
            shutil.copyfile(old_png_file,new_png_file)
        # 将图片变成yuv格式
        cmd='cd {} && ffmpeg -s 64x64 -pix_fmt yuv420p -y -r 10 -i %4d.png input.yuv'.format(tmp_dir)
        os.system(cmd)
        shutil.copyfile(os.path.join(tmp_dir,'input.yuv'),os.path.join(VVCdir,'input.yuv'))
        shutil.rmtree(tmp_dir)

        frames=len(os.listdir(png_dir))
        # 第二步:使用命令行运行EncoderAppStaticd
        cmd = "cd {} && ./EncoderAppStaticd  -c encoder_lowdelay_vtm.cfg -q {} -f {} > Enc_Out.txt".format(
                VVCdir, config['qp'], frames)
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
        makedirs(save_png_dir)
        # 第一步: 将bin解码为yuv
        command = "cd {} && ./DecoderAppStaticd -b str.bin -o output.yuv > Dec_Out.txt".format(
                VVCdir)
        os.system(command)
        shutil.copyfile(os.path.join(VVCdir,'output.yuv'),os.path.join(save_png_dir,'output.yuv'))
        # 第二步: 将yuv变为png图片序列
        command = "cd {} && ffmpeg -s 64x64 -pix_fmt yuv420p10le -y -i output.yuv %4d.png".format(
                save_png_dir)
        os.system(command)
        return
    
    for utt_id in utt_id_list:
        session = utt_id[4]
        png_dir=os.path.join(config['data_root'],'Session{}/face/{}'.format(session,utt_id))
        VVCdir='./VVCSoftware_VTM-VTM-15.0/encode_video_demo{}'.format(index)
        encode(png_dir,VVCdir)
        save_file='./trad_result/V/qp{}_index{}.txt'.format(config['qp'],index)
        get_result(utt_id,VVCdir,save_file)
        save_png_dir=os.path.join(config['trad_data_root'],'V/qp{}/{}'.format(config['qp'],utt_id))
        decode(VVCdir,save_png_dir)

def get_feature(config):

    # 抓取特征时需要的video_transform，在extract_feature方法中被使用
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)
    video_transform= transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        normTransform
    ])
    def extract_feature(model,img_paths):
        clip=[]
        model=model.to(device)
        for img_path in img_paths:
            img=Image.open(img_path).convert('RGB')
            clip.append(video_transform(img))
        clip = torch.stack(clip, 0).to(device)
        model.eval()
        with torch.no_grad():
            out=model(clip)
        if len(out.shape)==1:
            out=out.unsqueeze(0)
        # 将尺寸小于50的特征规整为(50,feature size)
        if out.shape[0]<50:
            pad = nn.ZeroPad2d(padding=(0, 0, 0, 50-out.shape[0]))
            out = pad(out)
        return out 

    extractor =DenseNet(growthRate=12, depth=100, reduction=0.5,bottleneck=True, nClasses=10,pre_train=True,device=device)
    all_utt_ids = get_all_utt_id(config)
    data_root=os.path.join(config['trad_data_root'],'V/qp{}'.format(config['qp']))
    feat_save_path = os.path.join(config['trad_feature_root'], "V/qp{}.h5".format(config['qp']))
    h5f = h5py.File(feat_save_path, 'w')
    print('start getting feature')
    video_dim=342
    for utt_id in tqdm(all_utt_ids):
        png_dir=os.path.join(data_root,utt_id)
        utt_face_pics = sorted(glob.glob(os.path.join(png_dir, '*.png')), key=lambda x: float(os.path.basename(x).split('.png')[0]))
        if len(utt_face_pics)!=0:
            utt_feats = extract_feature(extractor,utt_face_pics)
        else:
            utt_feats=[] 
        utt_feats=[a.cpu().numpy() for a in utt_feats]
        if len(utt_feats) != 0:
            utt_feats = np.array(utt_feats)
        else:
            print('{} missing face'.format(utt_id))
            utt_feats = np.zeros([50, video_dim])
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定模态遗失
        if 'Ses05F_script02_' in utt_id and utt_id[-4]=='M':
            print('{} missing video'.format(utt_id))
            utt_feats=np.zeros(utt_feats.shape)
        h5f[utt_id]=utt_feats
    h5f.close()

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
    # 添加多进程信息
    config['thread']=13
    # 创建文件夹
    save_dir_list = [config['trad_feature_root'],config['trad_data_root'],'trad_result']
    for save_dir in save_dir_list:
        for modality in ['A', 'V', 'L']:
            modality_dir = os.path.join(save_dir, modality)
            if not os.path.exists(modality_dir):
                os.makedirs(modality_dir)
    
    qp_list=[55,54,40,39,38]
    for qp in qp_list:
        # 添加传统编解码qp信息
        config['qp']=qp
        make_all_face(config)