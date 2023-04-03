import os
import pandas as pd

# 获得我们数据预处理方法与原作者数据预处理方法结果

modality_list=['avl','azz','zvz','zzl','avz','azl','zvl','total']
quality_list=[0,95,90,85,80]
def get_result(ours_root_dir,their_root_dir,save_csv_file):
    r'''获得我们数据预处理方法与原作者数据预处理方法结果, 存到csv文件中'''
    result=[]
    quality_index_map={0:0,95:5,90:10,85:15,80:20}
    for _ in range(25):
        result.append([])
    for quality in quality_list:
        result[quality_index_map[quality]]+=['{}'.format(quality)]*24
        for modality in modality_list:
            result[quality_index_map[quality]+1]+=['{}'.format(modality),'ours','their']
            ours_file=os.path.join(ours_root_dir,'result/compress{}/result_{}.tsv'.format(quality,modality))
            acc,uar,f1=get_metric(ours_file)
            result[quality_index_map[quality]+2]+=['acc',acc]
            result[quality_index_map[quality]+3]+=['uar',uar]
            result[quality_index_map[quality]+4]+=['f1',f1]
            their_file=os.path.join(their_root_dir,'result/compress{}/result_{}.tsv'.format(quality,modality))
            acc,uar,f1=get_metric(their_file)
            result[quality_index_map[quality]+2]+=[acc]
            result[quality_index_map[quality]+3]+=[uar]
            result[quality_index_map[quality]+4]+=[f1]
    df = pd.DataFrame(result)
    df.to_csv(save_csv_file,header=None,index=None)

def get_metric(file):
    with open(file,'r') as f:
        lines=f.readlines()
    acc,uar,f1=(float(a) for a in lines[-1].strip().split('\t'))
    return acc,uar,f1

if __name__=='__main__':
    get_result(r'D:\BaiduNetdiskDownload\BaiduNetdiskDownload\upload_result\ours\mmin_IEMOCAP_block_5_run0',
    r'D:\BaiduNetdiskDownload\BaiduNetdiskDownload\upload_result\their\mmin_IEMOCAP_block_5_run1',
    r'D:\BaiduNetdiskDownload\BaiduNetdiskDownload\ours_their.csv')