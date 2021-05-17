import torch
import torch.nn as nn
import torch.nn.functional as F

#### src: https://github.com/AlexiaJM/RelativisticGAN/blob/master/code/GAN_losses_iter.py
#### Create Encoder ####

import math, os
import torch.nn.utils.spectral_norm as spectral_norm

#### Source: https://github.com/mkisantal/backboned-unet/blob/master/backboned_unet/unet_backbone.py
class ResNet(nn.Module):

    """ U-Net downsampling block. """

    def __init__(self, in_channels, out_channels, downsample=True):
        super(ResNet, self).__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.MaxPool2d((2,2)) if downsample else None
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class Encoder(torch.nn.Module):
    def __init__(self, input_size, n_colors):
        super(Encoder, self).__init__()

        self.n_features = int(input_size/8)

        self.module1 = ResNet(n_colors, 64, downsample=False)
        self.module2 = ResNet(64, 128)
        self.module3 = ResNet(128, 256)
        self.module4 = ResNet(256, 512)
        
        self.dense = nn.Linear(512 * self.n_features * self.n_features, 128)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return self.tanh(x)

class Generator(nn.Module):
    def __init__(self, z_size, input_size, n_colors, spectral_G=False, Tanh_GD=False, no_batch_norm_G=False):
        super(Generator, self).__init__()

        self.z_size = z_size
        self.n_features = int(input_size/8)
        self.dense = torch.nn.Linear(128+self.z_size, 512 * self.n_features * self.n_features)

        if spectral_G:
            model = [spectral_norm(torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))]
            model += [nn.ReLU(True),
                spectral_norm(torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))]
            model += [nn.ReLU(True),
                spectral_norm(torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True))]
            model += [nn.ReLU(True),
                spectral_norm(torch.nn.Conv2d(64, n_colors, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.Tanh()]
        
        else:
            model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.Conv2d(64, n_colors, kernel_size=3, stride=1, padding=1, bias=True)]
        
        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, x, z):
        input = torch.cat((x, z), dim=1)
        output = self.dense(input.view(-1, 128+self.z_size))
        output = output.view(-1, 512, self.n_features, self.n_features)
        output = self.model(output)
        output = self.tanh(output)
        return output

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, n_colors, spectral_D=True, Tanh_GD=False, no_batch_norm_D=False):
        super(Discriminator, self).__init__()

        self.n_features = int(input_size/8)
        self.dense = nn.Linear(512 * self.n_features * self.n_features, 1)

        if spectral_D:
            model = [spectral_norm(nn.Conv2d(n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),

                spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
                torch.nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),

                spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),

                spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True)]
        else:
            model = [nn.Conv2d(n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.dense(self.model(input).view(-1, 512 * self.n_features * self.n_features)).view(-1)
        return output