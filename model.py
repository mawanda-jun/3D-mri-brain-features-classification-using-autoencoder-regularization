# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)
from blocks import *
from utils import *

import torch
import torch.nn as nn
from collections import OrderedDict
from group_norm import GroupNormalization


class Encoder(nn.Sequential):
    def __init__(self, in_features=1, out_features=256, input_side_dim=48):
        super(Encoder, self).__init__()
        next_padding = calc_same_padding(input_side_dim, 3, 1)
        out_dim_0 = input_side_dim
        out_dim_1 = calc_conv_shape(out_dim_0, 3, 0, 2)
        out_dim_2 = calc_conv_shape(out_dim_1, 3, 0, 2)
        out_dim_3 = calc_conv_shape(out_dim_2, 3, 0, 2)

        modules = [
            ('conv0', nn.Conv3d(in_features, out_channels=32, kernel_size=3, stride=1, padding=next_padding)),
            ('sp_drop0', nn.Dropout3d(0.2)),
            ('green0', GreenBlock(32, 32, out_dim_0)),
            ('downsize_0', nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)),
            # add padding to divide images by 2 exactly
            ('green10', GreenBlock(64, 64, out_dim_1)),
            ('green11', GreenBlock(64, 64, out_dim_1)),
            ('downsize_1', nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)),
            ('green20', GreenBlock(128, 128, out_dim_2)),
            ('green21', GreenBlock(128, 128, out_dim_2)),
            ('downsize_2', nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)),
            ('green30', GreenBlock(256, out_features, out_dim_3)),
            ('green31', GreenBlock(out_features, out_features, out_dim_3)),
            ('green32', GreenBlock(out_features, out_features, out_dim_3)),
            ('green33', GreenBlock(out_features, out_features, out_dim_3)),
        ]
        for m in modules:
            self.add_module(*m)


class Classifier(nn.Module):
    def __init__(self, in_features=256, num_features=5, input_side_dim=48):
        super(Classifier, self).__init__()
        out_dim_0 = calc_conv_shape(input_side_dim, 1, 0, 1)
        out_dim_1 = calc_conv_shape(out_dim_0, 1, 0, 1)
        out_dim_2 = calc_conv_shape(out_dim_1, 1, 0, 1)
        # print(out_dim_0, out_dim_1, out_dim_2)
        next_padding = calc_same_padding(out_dim_2, 3, 1)
        out_dim_3 = calc_conv_shape(out_dim_2, 3, next_padding, 1)

        self.to_ground_truth = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels=in_features, out_channels=128, kernel_size=(1, 1, 1), stride=1, padding=0)),
            ('green0', GreenBlock(128, 128, out_dim_0)),
            ('conv1', nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1), stride=1, padding=0)),
            ('green1', GreenBlock(64, 64, out_dim_1)),
            ('conv2', nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1, padding=0)),
            ('green2', GreenBlock(32, 32, out_dim_2)),
            (
            'conv3', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=next_padding)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        # print('Classifier has {} features'.format(32 * out_dim_3 ** 3))
        self.regressor = nn.Linear(in_features=32 * out_dim_3 ** 3, out_features=num_features)

    def forward(self, inputs):
        conv_out = self.to_ground_truth(inputs)
        return self.regressor(conv_out.view(conv_out.shape[0], -1))


class VAERegularization(nn.Module):
    def __init__(self, in_features=256, repr_dim=128, input_side_dim=6):
        super(VAERegularization, self).__init__()
        next_padding = calc_same_padding(input_side_dim, 3, 2)
        # VAE regularization
        self.reduce_dimension = nn.Sequential(OrderedDict([
            ('group_normR', GroupNormalization(in_features, groups=8)),
            ('reluR0', nn.ReLU(inplace=True)),
            ('convR0',
             nn.Conv3d(in_channels=in_features, out_channels=16, kernel_size=(3, 3, 3), stride=2,
                       padding=next_padding)),
        ]))
        out_dim = input_side_dim
        # print("out dim after VAE: {}".format(out_dim))
        # REPARAMETERIZATION TRICK (needs flattening)
        self.out_linear = nn.Linear(in_features=16 * out_dim ** 3, out_features=256)
        self.z_mean = nn.Linear(in_features=256, out_features=repr_dim)
        self.z_var = nn.Linear(in_features=256, out_features=repr_dim)
        self.reparameterization = Reparametrization()

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.out_linear(x.view(x.shape[0], -1))
        z_mean = self.z_mean(x)
        z_var = self.z_var(x)
        del x
        return self.reparameterization(z_mean, z_var), z_mean, z_var


class Decoder(nn.Module):
    def __init__(self, repr_dim=128, features_shape=(8, 3, 3, 3), num_channels=1, input_side_dim=6):
        super(Decoder, self).__init__()

        self.c, self.H, self.W, self.D = features_shape

        self.reshape_block = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=repr_dim, out_features=self.c*self.H*self.W*self.D)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        blue_padding = calc_same_padding(48, 3, 1)
        self.decode_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(self.c, 256, kernel_size=1, stride=1)),
            ('up1', nn.Upsample(scale_factor=2)),
            ('upgreen0', UpGreenBlock(256, 128, input_side_dim)),
            ('upgreen1', UpGreenBlock(128, 64, 12)),
            ('upgreen2', UpGreenBlock(64, 32, 24)),
            ('blue_block', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=blue_padding)),
            ('output_block', nn.Conv3d(in_channels=32, out_channels=num_channels, kernel_size=1, stride=1))
        ]))

    def forward(self, inputs):
        x = self.reshape_block(inputs)
        x = x.reshape([x.shape[0], self.c, self.H, self.W, self.D])
        x = self.decode_block(x)
        return x


class BrainClassifierVAE(nn.Module):
    def __init__(self, input_shape=(1, 48, 48, 48), num_features=16):
        super().__init__()
        c, H, W, D = input_shape
        self.repr_channel = 8  # c // 1, but the representation shrinks the information until 128, so I decided to keep it wider (so 8*3*3*3 ~ 128)
        self.repr_height = H // 16
        self.repr_width = W // 16
        self.repr_depth = D // 16
        self.input_shape = input_shape

        # ENCODING
        self.encoder = Encoder(in_features=c, out_features=256, input_side_dim=input_shape[1])
        # input_side_dim = 6

        # VAE regularization
        self.internal_representation = VAERegularization(in_features=256, repr_dim=128, input_side_dim=6)

        # DECODER
        self.decoder = Decoder(
            repr_dim=128,
            features_shape=(self.repr_channel, self.repr_height, self.repr_width, self.repr_depth),
            num_channels=c,
            input_side_dim=6
        )

        # CLASSIFICATION
        self.classifier = Classifier(in_features=256, num_features=num_features, input_side_dim=6)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        out_features = self.classifier(encoded)
        int_repr, z_mean, z_var = self.internal_representation(encoded)
        del encoded
        reconstructed_image = self.decoder(int_repr)
        return out_features, reconstructed_image, z_mean, z_var


if __name__ == '__main__':
    from losses import VAELoss

    bigboi = BrainClassifierVAE(input_shape=(1, 48, 48, 48), num_features=16).cuda()

    input = torch.rand((15, 1, 48, 48, 48)).cuda()
    target = torch.rand((15, 16)).cuda()

    out_features, reconstructed_image, z_mean, z_var = bigboi(input)

    lr = 1e-4
    weight_L2 = 0.1
    weight_KL = 0.1
    dice_e = 1e-8

    optim = torch.optim.AdamW(bigboi.parameters(), lr=1e-4)
    # Loss for features
    loss_mse = torch.nn.MSELoss()
    loss_mse_v = loss_mse(target, out_features)

    # Loss for VAE
    loss_vae = VAELoss(
        weight_KL=weight_KL,
        weight_L2=weight_L2
    )

    loss_vae_v = loss_vae(input, reconstructed_image, z_mean, z_var)

    print('loss_gt: {}'.format(loss_mse_v))
    print('loss_vae: {}'.format(loss_vae_v))

    print("features shape: {}".format(out_features.shape))
    print("reconstructed shape: {}".format(reconstructed_image.shape))
    print("z_mean: {} and z_var shapes: {}".format(z_mean.shape, z_var.shape))
