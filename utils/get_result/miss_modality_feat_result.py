import os
import pandas as pd

# 获得特征压缩状态下的结果

modality_list=['avl','azz','zvz','zzl','avz','azl','zvl']
quality_list=[0,95,90,85,80,75,70,65,60]

def get_feat_compared_result(root_dir,save_csv_file):
    r'''获得各个模态下的特征压缩率失真结果, 存到csv文件中'''
    result=[]
    quality_index_map={0:1,95:2,90:3,85:4,80:5,75:6,70:7,65:8,60:9}
    for _ in range(len(quality_index_map)+1):
        result.append([])
    for modality in modality_list:
        result[0]+=[modality,'kbps','acc','uar','f1']
        for quality in quality_list:
            kbps,acc,uar,f1=get_result(root_dir,modality,quality)
            result[quality_index_map[quality]]+=[quality,kbps,acc,uar,f1]
    df = pd.DataFrame(result)
    df.to_csv(save_csv_file,header=None,index=None)
    

def get_result(root_dir,modality,quality):
    r'''返回特定模态特定编码质量下的kbps, acc, uar, f1'''
    file_dir=os.path.join(root_dir,'{}/compressed_feat/{}/{}')
    bits=0
    sec=0
    for i in range(1,11):
        png_dir=file_dir.format(i,quality,modality)
        for file in os.listdir(png_dir):
            file=os.path.join(png_dir,file)
            bits+=os.path.getsize(file)*8
            index2=os.path.basename(file).find('-')
            sec+=float(os.path.basename(file)[:index2])
    kbps=bits/(1000*sec)
    result_file=os.path.join(root_dir,'result/compress{}/result_{}.tsv'.format(quality,modality))
    with open(result_file,'r') as f:
        lines=f.readlines()
    acc,uar,f1=(float(a) for a in lines[-1].strip().split('\t'))
    return kbps,acc,uar,f1


if __name__=='__main__':
    get_feat_compared_result(r'/home/haojun/docker/code/Graduation_Project/Graduation_Project_baseline/feat_result/ours/mmin_IEMOCAP_block_5_run5',
    r'./tmp.csv')