import torch
import os
import numpy as np
import cv2
from torch.autograd import Variable
from PIL import Image

def quantize_feature_validation(center_feature, n_bits=8):  # I consider numbers to be 255
    r'''input:
        center_feature: tensor, [batch, channel ,width ,height] 
    '''
    # data_shape = center_feature.data.shape
    center_flatten = center_feature.view(center_feature.numel())
    max_center_value, _ = center_flatten.max(0)
    min_center_value, _ = center_flatten.min(0)
    max_value_return = max_center_value.data.cpu().numpy()
    min_value_return = min_center_value.data.cpu().numpy()

    if torch.eq(min_center_value, max_center_value).all():
        max_center_value = max_center_value.add(10 ^ (-20)) #to avoid zero range
    V_minus_min_v = center_feature - min_center_value.expand_as(center_feature)
    range_center = max_center_value.add(-1 * min_center_value)
    range_center_inv = range_center.pow(-1)
    bit_range = 2 ** (n_bits) - 1
    range_center_inv_bit_range = bit_range * range_center_inv
    value_before_round = V_minus_min_v * range_center_inv_bit_range.expand_as(V_minus_min_v)
    quantized_value_255 = value_before_round.round()
    # Now we go all the way back to get the original value
    range_center_bit_range_inv = range_center_inv_bit_range.pow(-1)
    orig_before_add_min = quantized_value_255 * range_center_bit_range_inv.expand_as(quantized_value_255)
    center_recon = orig_before_add_min + min_center_value.expand_as(orig_before_add_min)
    return center_recon, quantized_value_255, max_value_return, min_value_return


def dump_feature_2D(feature, filename, max_range, min_range,png_jpg_flag,n_bits=8):
    r'''
    input:
        feature:np.ndarray, 形状是[channel, wight, height], 已经量化好的特征
    '''
    # PNG JPG FLAG , 0: PNG, 80:85:90:95 JPG
    if png_jpg_flag == 0:
        filename_write = filename + '.png'
    else:
        filename_write = filename + '.jpg'
    if not os.path.exists(os.path.dirname(filename_write)):
        os.makedirs(os.path.dirname(filename_write))

    # f_height是高，f_width是宽
    fch, f_height, f_width = feature.shape
    feature_vector = np.reshape(feature, (1, -1))
    feature_2D_width = int((2 ** np.ceil(1 / 2 * (np.log2(fch)))) * f_width)
    feature_2D_height = (((fch*f_width)//feature_2D_width)+1)*f_height
    # feature_2D_height = int((2 ** np.floor(1 / 2 * (np.log2(fch)))) * f_height)
    counter = -1
    feature_2D = np.zeros((feature_2D_height, feature_2D_width))
    # 填充feture 2D
    for i in range(0, feature_2D_height, f_height):
        for j in range(0, feature_2D_width, f_width):
            for k in range(i, i + f_height):
                for l in range(j, j + f_width):
                    counter += 1
                    if counter>=feature_vector.shape[1]:
                        break
                    feature_2D[k][l] = feature_vector[0][counter]

    if png_jpg_flag == 0:
        cv2.imwrite(filename_write, feature_2D)  #use different libraries for saving PNG/JPEG images
    else:
        obj = Image.fromarray(feature_2D)
        obj = obj.convert("L")
        obj.save(filename_write, format='JPEG', quality=png_jpg_flag)


    read_feature_2D = cv2.imread(filename_write,flags=cv2.IMREAD_GRAYSCALE)
    print (str(os.path.getsize(filename_write)))  #THIS GIVES IN BYTES!!!!!

    # change the image to tensor!
    channel_counter = -1
    # print (feature.shape)
    read_3D_feature = np.zeros(feature.shape)
    # print (read_3D_feature.shape)
    for i in range(0, feature_2D_height, f_height):
        for j in range(0, feature_2D_width, f_width):
            channel_counter += 1
            if channel_counter>=read_3D_feature.shape[0]:
                break
            for k in range(i, i + f_height):
                for l in range(j, j + f_width):
                    read_3D_feature[channel_counter][k - i][l - j] = read_feature_2D[k][l]
    bit_range = 2 ** (n_bits) - 1
    read_3D_feature = (read_3D_feature * (max_range - min_range) / bit_range) + min_range
    return Variable(torch.Tensor(read_3D_feature))


def quantize_feature_train(center_feature, n_bits=8):
    # r1 = -0.5
    # r2 = 0.5
    r1 = -0.002
    r2 = 0.002  # 2*0.002*256 = 1 it is the change in the 1/255 not 255 scale
    data_shape = center_feature.data.shape
    quant_error_t = (r1 - r2) * torch.rand(data_shape) + r2

    # quant_error = Variable(quant_error.cuda(), requires_grad=True)
    quant_error = quant_error_t.to(center_feature.device)
    # quant_error.cuda()
    center_recon = center_feature + quant_error
    return center_recon


def save_compressed_feat(feat_compress,save_path,epoch,quality):
        _,features_val_255, max_to_encode, min_to_encode = \
            quantize_feature_validation(feat_compress)
        features_val_255_numpy = features_val_255.data.cpu().numpy()
        _, _, f_height, f_width = features_val_255.shape
        features_val_255_numpy=features_val_255_numpy.reshape((-1,f_height,f_width))
        file_name= os.path.join(save_path,'epoch{}'.format(epoch))
        _ = dump_feature_2D (features_val_255_numpy,file_name,
            max_to_encode, min_to_encode,quality)

if __name__=='__main__':
    a=torch.randn((130,3,16,8))
    # cnt=1
    # for i in range(256):
    #     for j in range(2):
    #         for k in range(16):
    #             for q in range(8):
    #                 a[i][j][k][q]=cnt
    #                 cnt+=1
    save_compressed_feat(a,'a',0,0)