import os
import json
import shutil

def auto_train_CAP(args_dict):
    utt_fusion_train=True
    mmin_train=True
    cv_iter=range(1,11)
    type=args_dict['type']
    # 根据qp和kbps选择相应的raw feature
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = './data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))
    # 选择视频raw feature
    shutil.copyfile(os.path.join(config['trad_feature_root'],'V/qp{}.h5'.format(args_dict['qp'])),
                    os.path.join(config['feature_root'],'V/efficientface.h5'))
    # 选择音频raw feature
    shutil.copyfile(os.path.join(config['trad_feature_root'],'A/kbps{}.h5'.format(args_dict['kbps'])),
                    os.path.join(config['feature_root'],'A/comparE.h5'))
    shutil.copyfile(os.path.join(config['trad_feature_root'],'A/kbps{}_mean_std.h5'.format(args_dict['kbps'])),
                    os.path.join(config['feature_root'],'A/comparE_mean_std.h5'))
    # 选择文本raw feature
    shutil.copyfile(os.path.join(config['trad_feature_root'],'L/bert_large.h5'.format(args_dict['kbps'])),
                    os.path.join(config['feature_root'],'L/bert_large.h5'))
    
    # if os.path.exists(args_dict['checkpoints_dir']):
    #     shutil.rmtree(args_dict['checkpoints_dir'])
    # if os.path.exists(args_dict['log_dir']):
    #     shutil.rmtree(args_dict['log_dir'])
    # 命令行格式
    utt_fusion_cmd=('python train_baseline.py --dataset_mode=multimodal --model=utt_fusion'
        ' --gpu_ids={0[gpu_ids]} --modality=AVL --corpus_name=IEMOCAP'
        ' --log_dir={0[log_dir]} --checkpoints_dir={0[checkpoints_dir]} --print_freq=10' 
        ' --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size={0[embd_size]}'
        ' --embd_method_a=maxpool --V_type=efficientface --input_dim_v={0[input_dim_v]} --embd_method_v=maxpool'
        ' --L_type=bert_large --input_dim_l=1024'
        ' --output_dim=4 --cls_layers=128,128 --dropout_rate=0.3'
        ' --niter={0[niter]} --niter_decay=10 --in_mem --beta1=0.9'
        ' --batch_size={0[batch_size]} --lr=2e-4 --run_idx={0[run_idx]}'
        ' --name=CAP_utt_fusion --suffix=AVL_run{0[run_idx]}' 
        ' --has_test --cvNo={0[cvNo]} --feat_compress_size={0[feat_compress_size]}'
        ' --quality={0[quality]} --type trad --has_test')
    
    mmin_cmd=('python train_mmin.py --dataset_mode=multimodal_miss --model=mmin'
        ' --log_dir={0[log_dir]} --checkpoints_dir={0[checkpoints_dir]} --gpu_ids={0[gpu_ids]}'
        ' --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_method_a=maxpool'
        ' --V_type=efficientface --input_dim_v={0[input_dim_v]}  --embd_method_v=maxpool'
        ' --L_type=bert_large --input_dim_l=1024'
        ' --AE_layers=256,128,64 --n_blocks={0[n_blocks]} --num_thread=0 --corpus=IEMOCAP' 
        ' --ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0'
        ' --output_dim=4 --cls_layers=128,128 --dropout_rate=0.5'
        ' --niter={0[niter]} --niter_decay=30 --verbose --print_freq=10 --in_mem'
        ' --batch_size={0[batch_size]} --lr=2e-4 --run_idx={0[run_idx]} --weight_decay=1e-5'
        ' --name=mmin_IEMOCAP --suffix=block_{0[n_blocks]}_run{0[run_idx]} --has_test'
        ' --cvNo={0[cvNo]} --embd_size={0[embd_size]} --feat_compress_size={0[feat_compress_size]}'
        ' --quality={0[quality]} --type trad --has_test')
    
    for i in cv_iter:
        if args_dict['compress_flag']:
            mmin_cmd+=' --feat_compress'
            utt_fusion_cmd+=' --feat_compress'
        if args_dict['save_compress_pic']:
            mmin_cmd+=' --save_compress_pic'
            utt_fusion_cmd+=' --save_compress_pic'
        
        if utt_fusion_train:
            args_dict['cvNo']=i
            cmd=utt_fusion_cmd.format(args_dict)
            print(cmd)
            os.system(cmd)

        if mmin_train:
            mmin_cmd+=' --pretrained_path=./checkpoints/CAP_utt_fusion_AVL_run{0[run_idx]}'
            if args_dict['compress_flag']:
                mmin_cmd+='_feat'
            args_dict['cvNo']=i
            cmd=mmin_cmd.format(args_dict)
            print(cmd)
            os.system(cmd)
    
    
    os.makedirs('./trad_result/{}'.format(type),exist_ok=True)
    fusion_dir='CAP_utt_fusion_AVL_run{0[run_idx]}'.format(args_dict)
    mmin_dir='mmin_IEMOCAP_block_{0[n_blocks]}_run{0[run_idx]}'.format(args_dict)
    os.system('cp -rf {0[log_dir]}/{1}/ trad_result/{2}/'.format(args_dict,fusion_dir,type))
    os.system('cp -rf {0[log_dir]}/{1}/ trad_result/{2}/'.format(args_dict,mmin_dir,type))
    os.system('cp -rf {0[checkpoints_dir]}/{1}/ trad_result/{2}/'.format(args_dict,fusion_dir,type))
    os.system('cp -rf {0[checkpoints_dir]}/{1}/ trad_result/{2}/'.format(args_dict,mmin_dir,type))
    os.system('cd trad_result/{}/ && rm -rf */*/*.npy */*/*.pth'.format(type))

if __name__ =='__main__':
    exp_No=0
    gpu=1
    compress_flag=False
    type='trad'
    # 视频选择
    qp=56
    # 音频选择
    kbps=5
    args_dict={'qp':qp,'kbps':kbps,'run_idx':exp_No,'gpu_ids':gpu,'type':type,'embd_size':128,
        'feat_compress_size':'16,8','n_blocks':5,'quality':'0,95,90,85,80','niter':100,
        'compress_flag':compress_flag,'save_compress_pic':True,
        'checkpoints_dir':'./checkpoints','log_dir':'./logs','batch_size':128,
        'input_dim_v':342}
    auto_train_CAP(args_dict)