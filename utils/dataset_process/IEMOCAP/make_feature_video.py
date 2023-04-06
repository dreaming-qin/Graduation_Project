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
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import sys
current_directory = os.path.abspath(__file__)
for _ in range(4):
    current_directory=os.path.dirname(current_directory)
sys.path.append(current_directory)


from utils.dataset_process.tools.densenet import DenseNet


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
time_step=0.1

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def resize(location,size):
    r'''本论文中, 裁剪到的脸部尺寸大小都是64*64的, 所以这里设置裁剪的尺寸
    location是坐标'''
    top, right, bottom, left = location
    y_offset=size[1]-(bottom-top)
    x_offset=size[0]-(right-left)
    top-=y_offset//2
    bottom+=(y_offset+1)//2
    left-=x_offset//2
    right+=(x_offset+1)//2
    return top, right, bottom, left

def video_to_png(video_file,save_dir):
    makedirs(save_dir)
    #读取视频
    cap = cv2.VideoCapture(video_file)
    fps = round(cap.get(cv2.CAP_PROP_FPS)) #获取视频的帧率
    assert fps==30
    timestamp=1/fps
    step=round(time_step/timestamp)
    cnt=0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        if i%step==0 and cap.grab():
            _, frame = cap.retrieve()  #解码,并返回捕获的视频帧    
            #拼接图片保存路径
            cnt=timestamp*i
            newPath = os.path.join(save_dir,'{:.4f}.png'.format(cnt))
            #将图片按照设置格式，保存到文件
            cv2.imencode('.png', frame)[1].tofile(newPath)

def divide_png(png_dir,transcriptions_txt,save_dir):
    r'''保存的路径将会是{save_dir}/{根据transcriptions_txt决定的name},
    在这里将会裁剪主要测试人员的那一部分
    返回所有的图片存储路径, 列表形式
    '''
    def check_name(name):
        if not name.startswith('Ses') :
            return False
        try:
            int(name[-3:])
        except Exception:
            return False
        return True
    
    with open(transcriptions_txt,'r') as f:
        lines=f.readlines()
    png_file_list=sorted(os.listdir(png_dir),key=lambda x:float(x.split('.png')[0]))
    ans=[]
    for line in lines:
        line=line.strip().split(' ')
        if not check_name(line[0]) :
            print('unvalid name: {} in {}'.format(line[0].strip(),
                os.path.basename(transcriptions_txt)))
            continue
        # 属性参数
        name=line[0]
        type1,type2=name[5],name[-4]
        assert type1=='F' or type1=='M'
        assert type2=='F' or type2=='M'
        makedirs(os.path.join(save_dir,name))
        ans.append(os.path.join(save_dir,name))
        # 迭代要用到的参数
        start_time=float(line[1][1:line[1].find('-')])
        end_time=float(line[1][line[1].find('-')+1:-2])
        init_flag=False
        init_time=0
        for png_index in range(len(png_file_list)):
            png_file=os.path.join(png_dir,png_file_list[png_index])
            previx=float(png_file_list[png_index].split('.png')[0])
            if previx>end_time:
                break
            png_index+=1
            if previx<start_time:
                if png_index>=len(png_file_list):
                    break
                continue
            if not init_flag:
                init_flag=True
                init_time=previx
            img = cv2.imread(png_file)
            # 根据主要测试人员裁剪图片，裁剪坐标为[y0:y1, x0:x1]
            if type1==type2:
                cropped = img[:, 0:img.shape[1]//2]  
            else:
                cropped = img[:, img.shape[1]//2:]
            save_one=os.path.join(save_dir,name,'{:.4f}.png'.format(previx-init_time))  
            cv2.imwrite(save_one, cropped)
    return ans

def crop_face(png_dir,save_dir):
    r'''会存到{save_dir}/{os.path.basename(png_dir)}那
    '''
    # 需要进行人脸检测, 放在外面会报runtime错误
    import face_recognition
    save_dir=os.path.join(save_dir,os.path.basename(png_dir))
    makedirs(save_dir)

    face_locations_map={}
    face_locations_map[os.path.basename(png_dir)]={}
    miss_index=[]
    for png_file in os.listdir(png_dir):
        image = face_recognition.load_image_file(os.path.join(png_dir,png_file))
        # 基于hog机器学习模型进行人脸识别，不能使用gpu加速
        face_locations = face_recognition.face_locations(image)
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定裁剪黑色的图片
        if 'Ses05F_script02_' in png_dir and png_dir[-4]=='M':
            face_locations=[np.array([208,212,272,148])]
        if len(face_locations)!=1:
            face_locations_map[os.path.basename(png_dir)][png_file]=[]
            miss_index.append(png_file)
            continue
        face_locations_map[os.path.basename(png_dir)][png_file]=face_locations
        top, right, bottom, left = resize(face_locations[0],size=(64,64))
        

        # 提取人脸
        cropped = image[top:bottom, left:right]
        save_one=os.path.join(save_dir,png_file)  
        plt.imsave(save_one, cropped)
    if len(miss_index)!=0:
        print('missing index len of is {} in {}'.format(len(miss_index),os.path.basename(png_dir)))
    return face_locations_map

def deal_with_miss_face(png_dir,save_dir,face_location_map):
    r'''会存到{save_dir}/{face_location_map[key]}那
    {png_dir}/{face_location_map[key]}才是真正的图片文件夹

    如果在相同文件夹有照片, 则用该文件夹内所有照片的坐标均值做坐标
    如果文件夹内没有照片, 就用邻近文件夹的所有照片的坐标均值做坐标
    '''
    # 需要进行人脸检测, 放在外面会报runtime错误
    import face_recognition
    def get_dir_avg(face_location_map):
        avg_map={}
        for key1 in face_location_map:
            sum=[]
            for _,values in face_location_map[key1].items():
                if len(values)!=0:
                    for val2 in values:
                        sum.append(val2)
            if len(sum)!=0:
                avg=np.mean(sum,axis=0)
                avg_map[key1]=avg.astype(int)
        return avg_map
    
    def get_sorted_key(face_location_map,avg_map):
        sorted_Mlist=[]
        sorted_Flist=[]
        for key1 in face_location_map:
            if key1 not in avg_map:
                continue
            assert key1[-4]=='F' or key1[-4]=='M'
            if key1[-4]=='F':
                sorted_Flist.append(key1)
            elif key1[-4]=='M':
                sorted_Mlist.append(key1)
        return sorted(sorted_Mlist),sorted(sorted_Flist)
    
    def fix_missing(png_dir,save_dir,face_location_map,sorted_list,avg_map):
        # 先处理在相同文件夹有照片的情况
        for key1 in face_location_map:
            if key1 in avg_map:
                for key2,values2 in face_location_map[key1].items():
                    if len(values2)==0:
                        png_file=os.path.join(png_dir,key1,key2)
                        save_file=os.path.join(save_dir,key1,key2)
                        save_pic(avg_map[key1],save_file,png_file)
        # 再处理文件夹没有照片的情况
        sorted_index=0
        for key1 in face_location_map:
            if key1 not in avg_map:
                if int(key1[-3:])>int(sorted_list[sorted_index][-3:]) and\
                    sorted_index<len(sorted_list)-1:
                    sorted_index+=1
                for key2 in face_location_map[key1]: 
                    png_file=os.path.join(png_dir,key1,key2)
                    save_file=os.path.join(save_dir,key1,key2)
                    save_pic(avg_map[sorted_list[sorted_index]],save_file,png_file)
    
    def save_pic(locations,save_file,png_file):
        top, right, bottom, left = resize(locations,size=(64,64))
        image = face_recognition.load_image_file(png_file)
        # 提取人脸
        cropped = image[top:bottom, left:right]
        plt.imsave(save_file, cropped)
    
    avg_map=get_dir_avg(face_location_map)
    M_list,F_list=get_sorted_key(face_location_map,avg_map)
    fix_missing(png_dir,save_dir,face_location_map,M_list,avg_map)
    fix_missing(png_dir,save_dir,face_location_map,F_list,avg_map)

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

def _make_all_face(config):
    r'''函数内部调用'''
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
        if len(out.shape)==1:
            out=out.unsqueeze(0)
        # 将尺寸小于50的特征规整为(50,feature size)
        if out.shape[0]<50:
            pad = nn.ZeroPad2d(padding=(0, 0, 0, 50-out.shape[0]))
            out = pad(out)
        # else:
        #     out = out[:50, :]
        return out 

    extractor =DenseNet(growthRate=12, depth=100, reduction=0.5,bottleneck=True, nClasses=10,pre_train=True,device=device)
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feat_feature_root'], 'raw', "V", "raw_efficientface.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(all_utt_ids):
        sess_id = utt_id[4]
        face_dir = os.path.join(config['data_root'],'Session{}/face/{}'.format(sess_id,utt_id))
        utt_face_pics = sorted(glob.glob(os.path.join(face_dir, '*.png')), key=lambda x: float(x.split('/')[-1].split('.png')[0]))
        utt_feats = []
        # 抓语义级别的帧
        feat = extract_feature(extractor,utt_face_pics)
        utt_feats=[a.cpu().numpy() for a in feat]
        if len(utt_feats) != 0:
            utt_feats = np.array(utt_feats)
        else:
            print('missing face')
            utt_feats = np.zeros([1, 342])
        # 对于Ses05F_script02_*.avi，男方视频是黑色的，直接认定模态遗失
        if 'Ses05F_script02_' in utt_id and utt_id[-4]=='M':
            print('{} missing video'.format(utt_id))
            utt_feats=np.zeros(utt_feats.shape)
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = utt_feats
    h5f.close()

def make_all_efficientface(config):
    r'''供外部函数调用'''
    data_root=config['data_root']
    # get face pic
    session_list=[1,2,3,4,5]
    for session in session_list:
        # session=5
        video_dir=os.path.join(data_root,'Session{}/dialog/avi/DivX'.format(session))
        video_file_list=os.listdir(video_dir)
        for video_file in video_file_list:
            # video_file='Ses05F_script02_2.avi'
            if not video_file.endswith('.avi'):
                continue
            print('start get face: Session{} {}'.format(session,video_file))
            name=video_file.split('.')[0]
            video_to_png(os.path.join(video_dir,video_file),
                os.path.join(data_root,'Session{}/face/tmp/{}'.format(session,name)))
            png_dir=divide_png(os.path.join(data_root,'Session{}/face/tmp/{}'.format(session,name)),
                os.path.join(data_root,'Session{}/dialog/transcriptions/{}.txt'.format(session,name)),
                os.path.join(data_root,'Session{}/face/tmp'.format(session)))
            face_locations_map={}
            for abc in png_dir:
                map2=crop_face(abc,os.path.join(data_root,'Session{}/face'.format(session)))
                face_locations_map.update(map2)
            deal_with_miss_face(os.path.join(data_root,'Session{}/face/tmp'.format(session)),
                os.path.join(data_root,'Session{}/face'.format(session)),
                face_locations_map)
            shutil.rmtree(os.path.join(data_root,'Session{}/face/tmp'.format(session)))

    # make raw feat record with timestamp
    _make_all_face(config)

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
    data_root=config['data_root']
    # get face pic
    session_list=[1,2,3,4,5]
    # session_list=[3]
    # for session in session_list:
    #     video_dir=os.path.join(data_root,'Session{}/dialog/avi/DivX'.format(session))
    #     video_file_list=os.listdir(video_dir)
    #     for video_file in video_file_list:
    #         # video_file='Ses03M_impro07.avi'
    #         if not video_file.endswith('.avi'):
    #             continue
    #         print('start get face: Session{} {}'.format(session,video_file))
    #         name=video_file.split('.')[0]
    #         video_to_png(os.path.join(video_dir,video_file),
    #             os.path.join(data_root,'Session{}/face/tmp/{}'.format(session,name)))
    #         png_dir=divide_png(os.path.join(data_root,'Session{}/face/tmp/{}'.format(session,name)),
    #             os.path.join(data_root,'Session{}/dialog/transcriptions/{}.txt'.format(session,name)),
    #             os.path.join(data_root,'Session{}/face/tmp'.format(session)))
    #         face_locations_map={}
    #         for abc in png_dir:
    #             map2=crop_face(abc,os.path.join(data_root,'Session{}/face'.format(session)))
    #             face_locations_map.update(map2)
    #         deal_with_miss_face(os.path.join(data_root,'Session{}/face/tmp'.format(session)),
    #             os.path.join(data_root,'Session{}/face'.format(session)),
    #             face_locations_map)
    #         shutil.rmtree(os.path.join(data_root,'Session{}/face/tmp'.format(session)))

    # make raw feat record with timestamp
    _make_all_face(config)