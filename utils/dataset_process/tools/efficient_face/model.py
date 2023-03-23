import torch
import torch.nn as nn
import warnings
from PIL import Image

from utils.dataset_process.tools.efficient_face.modulator import Modulator
from  utils.dataset_process.tools.efficient_face.efficientface import LocalFeatureExtractor, InvertedResidual
from  utils.dataset_process.tools.efficient_face import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        init_feature_extractor(self,('/home/haojun/docker/code/Graduation_Project/'
        'Graduation_Project_baseline/utils/dataset_process/tools/efficient_face/'
        'EfficientFace_Trained_on_AffectNet7.pth'))
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) #global average pooling
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        warnings.warn('path is None! Initializing {} failed'.format(str(model.__class__)))
        return
    checkpoint = torch.load(path, map_location=device)
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict( pre_trained_dict, strict=False)
    # model.load_state_dict( torch.load(path, map_location=torch.device('cpu')), strict=False)


if __name__=='__main__':
    visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024])

    
    video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(255)])
    img=Image.open('a.png').convert('RGB')
    img=img.resize((224,224))
    img.save('./b.png')
    video_transform.randomize_parameters()
    clip = [video_transform(img) ,video_transform(img)]
    clip = torch.stack(clip, 0)
    ccc=visual_model(clip)   
    # conv1._modeules.'0'.weight
    a=1