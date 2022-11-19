import torch
import numpy as np
import os
import torchvision
from torch import nn
from torchvision.transforms import transforms

conv_layer = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.encoder = self.make_conv_layers(conv_layer['E'])
        self.decoder = self.make_deconv_layers(conv_layer['D'])

        self.net_params_path = './gen_modelWeights0090'
        self.net_params_pathDir = os.listdir(self.net_params_path)
        self.net_params_pathDir.sort()
        self.mymodules = nn.ModuleList([
            self.deconv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])
        self.cfg = cfg
    def upsampling(self, x):
        m = nn.UpsamplingBilinear2d(size=[self.cfg.image_h, self.cfg.image_w])
        x = m(x)
        return x

    def conv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def deconv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def relu(self, inplace=True):  # Change to True?
        return nn.ReLU(inplace)

    def maxpool2d(self, ):
        return nn.MaxPool2d(2)

    def make_conv_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [self.maxpool2d()]
            else:
                conv = self.conv2d(in_channels, v)
                layers += [conv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_deconv_layers(self, cfg):
        layers = []
        in_channels = 512
        for v in cfg:
            if v == 'U':
                layers += [nn.Upsample(scale_factor=2)]
            else:
                deconv = self.deconv2d(in_channels, v)
                layers += [deconv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        x = self.encoder[5](x)
        x = self.encoder[6](x)
        x = self.encoder[7](x)
        x = self.encoder[8](x)
        x = self.encoder[9](x)
        x = self.encoder[10](x)
        x = self.encoder[11](x)
        x = self.encoder[12](x)
        x = self.encoder[13](x)
        x = self.encoder[14](x)
        x = self.encoder[15](x)
        x = self.encoder[16](x)
        x = self.encoder[17](x)
        x = self.encoder[18](x)
        x = self.encoder[19](x)
        x = self.encoder[20](x)
        x = self.encoder[21](x)
        x = self.encoder[22](x)
        x = self.encoder[23](x)
        x = self.encoder[24](x)
        x = self.encoder[25](x)
        x = self.encoder[26](x)
        x = self.encoder[27](x)
        x = self.encoder[28](x)
        x = self.encoder[29](x)
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        x = self.decoder[5](x)
        x = self.decoder[6](x)
        x = self.decoder[7](x)
        x = self.decoder[8](x)
        x = self.decoder[9](x)
        x = self.decoder[10](x)
        x = self.decoder[11](x)
        x = self.decoder[12](x)
        x = self.decoder[13](x)
        x = self.decoder[14](x)
        x = self.decoder[15](x)
        x = self.decoder[16](x)
        x = self.decoder[17](x)
        x = self.decoder[18](x)
        x = self.decoder[19](x)
        x = self.decoder[20](x)
        x = self.decoder[21](x)
        x = self.decoder[22](x)
        x = self.decoder[23](x)
        x = self.decoder[24](x)
        x = self.decoder[25](x)
        x = self.decoder[26](x)
        x = self.decoder[27](x)
        f10 = self.decoder[28](x)
        # x = self.decoder[29](f10)
        f10 = self.upsampling(f10).data
        return f10


class Feature_Extrator():
    def __init__(self, cfg):
        self.net = self.gen_SalGAN(cfg).to(cfg.device)
        self.cfg = cfg

        self.m = nn.BatchNorm2d(cfg.feature_dim, affine=True).to(cfg.device)
        self.pool2d = nn.AvgPool2d(cfg.patch_size)
        self.transform = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def gen_SalGAN(self, cfg):
        net = Generator(cfg)
        params = net.state_dict()
        n1 = 0
        pretrained_dict = {}
        for k, v in params.items():
            single_file_name = net.net_params_pathDir[n1]
            single_file_path = os.path.join(net.net_params_path, single_file_name)
            pa = np.load(single_file_path)
            pa = torch.from_numpy(pa)
            pretrained_dict[k] = pa
            n1 += 1
        params.update(pretrained_dict)
        net.load_state_dict(params)
        return net

    def get_features_seqs(self, imgs, sals):
        with torch.no_grad():
            images = self.transform(imgs).float()
            features = self.net(images)
            # print(features.shape)
            # features = Variable(torch.cat((images1, features), 1))
            features = self.m(features)  # [B, C ,H ,W]
            sals = sals.unsqueeze(1).expand(imgs.size(0), self.cfg.feature_dim, features.size(2), features.size(3))
            features = features * sals
            features = self.pool2d(features)
            enc_inputs = features.flatten(2).permute(0, 2, 1)  # [B, H ,W, C]
            return enc_inputs
