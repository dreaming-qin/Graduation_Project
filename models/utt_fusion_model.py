
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
import sys

from utils.feature_compress import quantize_feature_train,quantize_feature_validation
from loss.loss import FeatureCompressLoss,DynamicWeightedLoss

class UttFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size', default=128, type=int, help='audio/text/visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.feat_compress_size=list(map(lambda x: int(x), opt.feat_compress_size.split(',')))
        self.feat_compress_flag=opt.feat_compress

        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        if self.feat_compress_flag:
            self.loss_names.append('feat_compress')
        self.modality = opt.modality
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size * int("A" in self.modality) + \
                         opt.embd_size * int("V" in self.modality) + \
                         opt.embd_size * int("L" in self.modality)
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size, embd_method=opt.embd_method_a)
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size, opt.embd_method_v)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            if self.feat_compress_flag:
                self.criterion_feat_compress=FeatureCompressLoss()
            self.criterion_dynamic_weight=DynamicWeightedLoss(2 if self.feat_compress_flag else 1)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]+\
                [{'params':p} for p in self.criterion_dynamic_weight.parameters()]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in self.modality:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.modality:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.modality:
            self.visual = input['V_feat'].float().to(self.device)
        
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            final_embd.append(self.feat_A)

        if 'L' in self.modality:
            self.feat_L = self.netL(self.lexical)
            final_embd.append(self.feat_L)
        
        if 'V' in self.modality:
            self.feat_V = self.netV(self.visual)
            final_embd.append(self.feat_V)
        self.final_embd=final_embd

        # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)
        length,height=self.feat_compress_size[0],self.feat_compress_size[1]
        self.feat_compress=torch.stack((self.feat_A.reshape(-1,length,height),
            self.feat_V.reshape(-1,length,height),
            self.feat_L.reshape(-1,length,height)),
            dim=1)
        # 模拟量化误差
        if self.isTrain:
            self.feat=quantize_feature_train(self.feat)
        else:
            self.feat,_,_,_=quantize_feature_validation(self.feat)
        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        losses_list=[('cla',self.loss_CE)]
        # self.feat_compress=torch.tensor(self.final_embd)
        if self.feat_compress_flag:
            self.loss_feat_compress=self.criterion_feat_compress(self.feat_compress)
            losses_list.append(('reg',self.loss_feat_compress))
        loss=self.criterion_dynamic_weight(losses_list)
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
