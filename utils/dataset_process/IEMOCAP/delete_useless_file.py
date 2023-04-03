import os
import json


r'''IEMOCAP数据集中有类似于._.DS_Store的文件, 对其进行删除'''

def delete_file(data_root):
    dir_list=os.listdir(data_root)
    for dir in dir_list:
        dir=os.path.join(data_root,dir)
        if os.path.isdir(dir):
            delete_file(dir)
        elif os.path.isfile(dir):
            if os.path.basename(dir).startswith('._'):
                os.remove(dir)
        else:
            print('{} is not a file or a directory'.format(dir))
    pass



if __name__=='__main__':
    # load config
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../../..', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    delete_file(config['data_root'])
