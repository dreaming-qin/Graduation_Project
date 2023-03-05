
import numpy as np
import math

def generate_DCT_matries(dim)-> np.ndarray:
    r'''获得形状为dim*dim的DCT变换矩阵
    output: 格式为numpy.ndarray, 形状为dim*dim
    '''
    A=np.zeros((dim,dim))#生成0矩阵
    for i in range(dim):
        for j in range(dim):
            if(i == 0):
                x=math.sqrt(1/dim)
            else:
                x=math.sqrt(2/dim)
            A[i][j]=x*math.cos(math.pi*(j+0.5)*i/dim)#与维数相关
    return A

    