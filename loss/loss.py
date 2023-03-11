import torch.nn as nn
import torch
import warnings

from utils.DCT import generate_DCT_matries


class FeatureCompressLoss(nn.Module):
    r'''用于特征压缩的Loss函数
    参考文献: S. R. Alvar and I. V. Bajić, "Multi-task learning with compressible features for Collaborative Intelligence".
        accepted for presentation at IEEE ICIP'19, Taipei, Taiwan, Sep. 2019.
    '''
    def __init__(self) -> None:
        r'''初始化DPCM, DCT矩阵
        input: 
            shape: 一个二元组, 指定压缩时一个特征的height和weight
        '''
        super().__init__()
        self.DHH=None
        self.DWW=None
        self.Mc=None
        self.Mr=None


    def forward(self,feature:torch.Tensor):
        r'''
        input: 
            feature: 类型为tensor, 形状为[batch_szei,channel, height, weight ]
        ouput: 类型为tensor, 输出特征压缩损失值
        '''
        _,f_channel,f_height,f_width=feature.shape
        if self.DHH is None:
            DHH=torch.eye(f_height)
            for i in range(f_height-1):
                DHH[i][i+1]=-1
            DWW=torch.eye(f_width)
            for i in range(f_width-1):
                DWW[i][i+1]=-1
            Mc=torch.Tensor(generate_DCT_matries(f_height))
            Mr=torch.Tensor(generate_DCT_matries(f_width))
            device=feature.device
            self.DHH=DHH.to(device)
            self.DWW=DWW.to(device)
            self.Mc=Mc.to(device)
            self.Mr=Mr.to(device)
        fx=torch.matmul(feature,self.DWW)
        fy=torch.matmul(self.DHH.T,feature)
        Z=0.5*(fx+fy)
        T=torch.matmul(self.Mc,Z)
        T=torch.matmul(T,self.Mr.T)
        return torch.abs(T).sum()/(f_channel*f_height*f_width)


class DynamicWeightedLoss(nn.Module):
    r'''动态调整权重的loss函数
    参考文献: Multi-Task Learning Using Uncertainty to Weigh Losses 
        for Scene Geometry and Semantics
    参考网址: https://blog.csdn.net/u010212101/article/details/115701136
    '''
    def __init__(self):
        super(DynamicWeightedLoss, self).__init__()

    def forward(self, x_list):
        r'''x_list中元素形式如下: (name, loss)
            name类型为str, 用来判断是回归还是分类, 为reg是回归, 为cla是分类
            loss类型为tensor, 形状为(1,), 是已经计算好的单个任务的Loss值
        '''
        if self.params is None:
            num=len(x_list)
            params = torch.zeros(num, requires_grad=True)
            self.params = torch.nn.Parameter(params)

        loss_sum = 0
        for i, (name,loss) in enumerate(x_list):
            if name=='reg':
                loss_sum += torch.exp(-self.params[i]) * loss+ self.params[i]
            elif name=='cla':
                loss_sum +=2*torch.exp(-self.params[i]) * loss + self.params[i]
            else:
                warnings.warn('warning: name is not either cla or cla')
                loss_sum += torch.exp(-self.params[i]) * loss+ self.params[i]
        return loss_sum

