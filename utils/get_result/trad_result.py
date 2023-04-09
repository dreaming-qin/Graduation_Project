
import os
import pandas as pd

r'''获得传统编解码和特征编解码的对比结果中的传统编解码结果
特征编解码结果在miss_modality_feat_result.py中
'''



# 获得特征压缩状态下的结果

modality_list=['avl','azz','zvz','zzl','avz','azl','zvl']

def get_trad_result(root_dir,save_csv_file,qp,audio_kbps):
    r'''获得各个模态下的特征压缩率失真结果, 存到csv文件中'''
    result=[]
    for _ in range(2):
        result.append([])
    for modality in modality_list:
        result[0]+=[modality,'kbps','acc','uar','f1']
        kbps,acc,uar,f1=get_result(root_dir,modality,qp,audio_kbps)
        result[1]+=['qp{}+kbps{}'.format(qp,audio_kbps),kbps,acc,uar,f1]
    df = pd.DataFrame(result)
    df.to_csv(save_csv_file,header=None,index=None,mode='a')
    

def get_result(root_dir,modality,qp,audio_kbps):
    r'''返回特定模态特定编码质量下的kbps, acc, uar, f1'''
    kbps=0
    # 获得视频kbps
    if 'v' in modality:
        dir_name='./trad_result/V/kbps.txt'
        with open(dir_name,'r')as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip().split(',')
            if line[0]=='{}'.format(qp):
                kbps+=float(line[1])
                break
    # 获得音频kbps
    if 'a' in modality:
        dir_name='./trad_result/A/kbps.txt'
        with open(dir_name,'r')as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip().split(',')
            if line[0]=='{}'.format(audio_kbps):
                kbps+=float(line[1])
                break
    # 获得文本kbps
    if 'l' in modality:
        dir_name='./trad_result/L/kbps.txt'
        with open(dir_name,'r')as f:
            lines=f.readlines()
        kbps+=float(lines[-1].strip())
    result_file=os.path.join(root_dir,'result/compress{}/result_{}.tsv'.format(0,modality))
    with open(result_file,'r') as f:
        lines=f.readlines()
    acc,uar,f1=(float(a) for a in lines[-1].strip().split('\t'))
    return kbps,acc,uar,f1


if __name__=='__main__':
    get_trad_result(r'/home/haojun/docker/code/Graduation_Project/Graduation_Project_baseline/trad_result/trad/mmin_IEMOCAP_block_5_run0',
    r'./tmp.csv',
    qp=56,audio_kbps=5)