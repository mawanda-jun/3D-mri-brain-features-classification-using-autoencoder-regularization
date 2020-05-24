# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)
from blocks import *
from utils import *

import torch
import torch.nn as nn
from collections import OrderedDict


# from group_norm import GroupNormalization


class Encoder(nn.Sequential):
    def __init__(self, in_features=1, input_side_dim=48, model_depth=32):
        super(Encoder, self).__init__()
        next_padding = calc_same_padding(input_side_dim, 3, 1)
        out_dim_0 = input_side_dim
        out_dim_1 = calc_conv_shape(out_dim_0, 3, 0, 2)
        out_dim_2 = calc_conv_shape(out_dim_1, 3, 0, 2)
        out_dim_3 = calc_conv_shape(out_dim_2, 3, 0, 2)
        # Define depth of model
        init_d = model_depth
        modules = [
            ('conv0', nn.Conv3d(in_features, init_d, kernel_size=3, stride=1, padding=next_padding)),
            ('sp_drop0', nn.Dropout3d(0.2)),
            ('green0', GreenBlock(init_d, init_d, out_dim_0)),
            ('downsize_0', nn.Conv3d(init_d, init_d * 2, kernel_size=3, stride=2, padding=1)),
            # add padding to divide images by 2 exactly
            ('green10', GreenBlock(init_d * 2, init_d * 2, out_dim_1)),
            ('green11', GreenBlock(init_d * 2, init_d * 2, out_dim_1)),
            ('downsize_1', nn.Conv3d(init_d * 2, init_d * 4, kernel_size=3, stride=2, padding=1)),
            ('green20', GreenBlock(init_d * 4, init_d * 4, out_dim_2)),
            ('green21', GreenBlock(init_d * 4, init_d * 4, out_dim_2)),
            ('downsize_2', nn.Conv3d(init_d * 4, init_d * 8, kernel_size=3, stride=2, padding=1)),
            ('green30', GreenBlock(init_d * 8, init_d * 8, out_dim_3)),
            ('green31', GreenBlock(init_d * 8, init_d * 8, out_dim_3)),
            ('green32', GreenBlock(init_d * 8, init_d * 8, out_dim_3)),
            ('green33', GreenBlock(init_d * 8, init_d * 8, out_dim_3)),
        ]
        for m in modules:
            self.add_module(*m)


class Classifier(nn.Module):
    def __init__(self, num_classes=5, input_side_dim=6, model_depth=32):
        super(Classifier, self).__init__()
        out_dim_0 = calc_conv_shape(input_side_dim, 1, 0, 1)
        out_dim_1 = calc_conv_shape(out_dim_0, 1, 0, 1)
        out_dim_2 = calc_conv_shape(out_dim_1, 1, 0, 1)
        # print(out_dim_0, out_dim_1, out_dim_2)
        next_padding = calc_same_padding(out_dim_2, 3, 1)
        out_dim_3 = calc_conv_shape(out_dim_2, 3, next_padding, 1)
        self.to_ground_truth = nn.Sequential(OrderedDict([
            ('conv0',
             nn.Conv3d(in_channels=model_depth * 8, out_channels=model_depth * 4, kernel_size=(1, 1, 1), stride=1,
                       padding=0)),
            ('green0', GreenBlock(model_depth * 4, model_depth * 4, out_dim_0)),
            ('conv1',
             nn.Conv3d(in_channels=model_depth * 4, out_channels=model_depth * 2, kernel_size=(1, 1, 1), stride=1,
                       padding=0)),
            ('green1', GreenBlock(model_depth * 2, model_depth * 2, out_dim_1)),
            ('conv2', nn.Conv3d(in_channels=model_depth * 2, out_channels=model_depth, kernel_size=(1, 1, 1), stride=1,
                                padding=0)),
            ('green2', GreenBlock(model_depth, model_depth, out_dim_2)),
            ('conv3',
             nn.Conv3d(in_channels=model_depth, out_channels=model_depth, kernel_size=(3, 3, 3), stride=1,
                       padding=next_padding)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        # print('Classifier has {} features'.format(model_depth * out_dim_3 ** 3))
        self.regressor = nn.Linear(in_features=model_depth * out_dim_3 ** 3, out_features=num_classes)

    def forward(self, inputs):
        conv_out = self.to_ground_truth(inputs)
        return self.regressor(conv_out.view(conv_out.shape[0], -1))


class VAERegularization(nn.Module):
    def __init__(self, repr_dim=128, input_side_dim=6, model_depth=32):
        super(VAERegularization, self).__init__()
        # VAE regularization
        self.reduce_dimension = nn.Sequential(OrderedDict([
            # ('group_normR', GroupNormalization(in_features, groups=8)),
            ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.ReLU(inplace=True)),
            ('convR0',
             nn.Conv3d(in_channels=model_depth * 8, out_channels=model_depth // 2, kernel_size=(3, 3, 3), stride=2,
                       padding=1)),
        ]))
        out_dim = 3
        # print("out dim after VAE: {}".format(out_dim))
        # REPARAMETERIZATION TRICK (needs flattening)
        self.out_linear = nn.Linear(in_features=(model_depth // 2) * out_dim ** 3, out_features=model_depth * 8)
        self.z_mean = nn.Linear(in_features=model_depth * 8, out_features=repr_dim)
        self.z_var = nn.Linear(in_features=model_depth * 8, out_features=repr_dim)
        self.reparameterization = Reparametrization()

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.out_linear(x.view(x.shape[0], -1))
        z_mean = self.z_mean(x)
        z_var = self.z_var(x)
        del x
        return self.reparameterization(z_mean, z_var), z_mean, z_var


class Decoder(nn.Module):
    def __init__(self, repr_dim=128, model_depth=32, num_channels=1, input_side_dim=3):
        super(Decoder, self).__init__()
        self.model_depth = model_depth
        self.input_side_dim = input_side_dim
        self.reshape_block = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=repr_dim, out_features=(model_depth // 2) * input_side_dim ** 3)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        blue_padding = calc_same_padding(48, 3, 1)
        self.decode_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(model_depth // 2, model_depth * 8, kernel_size=1, stride=1)),
            ('up1', nn.Upsample(scale_factor=2)),
            ('upgreen0', UpGreenBlock(model_depth * 8, model_depth * 4, input_side_dim)),
            ('upgreen1', UpGreenBlock(model_depth * 4, model_depth * 2, 12)),
            ('upgreen2', UpGreenBlock(model_depth * 2, model_depth, 24)),
            ('blue_block', nn.Conv3d(in_channels=model_depth, out_channels=model_depth, kernel_size=3, stride=1,
                                     padding=blue_padding)),
            ('output_block', nn.Conv3d(in_channels=model_depth, out_channels=num_channels, kernel_size=1, stride=1))
        ]))

    def forward(self, inputs):
        x = self.reshape_block(inputs)
        x = x.reshape([x.shape[0], self.model_depth // 2, self.input_side_dim, self.input_side_dim, self.input_side_dim])
        x = self.decode_block(x)
        return x


class BrainClassifierVAE(nn.Module):
    def __init__(self, in_channels=1, input_side_dim=48, num_classes=16, model_depth=32):
        super().__init__()

        # ENCODING
        self.encoder = Encoder(in_features=in_channels, input_side_dim=input_side_dim, model_depth=model_depth)
        # input_side_dim = 6

        # VAE regularization
        self.internal_representation = VAERegularization(repr_dim=128, input_side_dim=6, model_depth=model_depth)

        # DECODER
        # The internal representation shrinks the dimension by a factor of 2
        self.decoder = Decoder(
            repr_dim=128,
            model_depth=model_depth,
            num_channels=in_channels,
            input_side_dim=3
        )

        # CLASSIFICATION
        self.classifier = Classifier(num_classes=num_classes, input_side_dim=6, model_depth=model_depth)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        out_features = self.classifier(encoded)
        int_repr, z_mean, z_var = self.internal_representation(encoded)
        del encoded
        reconstructed_image = self.decoder(int_repr)
        return out_features, reconstructed_image, inputs, z_mean, z_var


if __name__ == '__main__':
    from losses import VAELoss

    bigboi = BrainClassifierVAE(in_channels=53, input_side_dim=48, num_classes=8, model_depth=32).cuda()

    input = torch.rand((10, 53, 48, 48, 48)).cuda()
    target = torch.rand((10, 8)).cuda()

    lr = 1e-4
    weight_L2 = 0.1
    weight_KL = 0.1
    dice_e = 1e-8

    optim = torch.optim.AdamW(bigboi.parameters(), lr=1e-4)
    # Loss for features
    loss_mse = torch.nn.MSELoss()
    # Loss for VAE
    loss_vae = VAELoss(
        weight_KL=weight_KL,
        weight_L2=weight_L2
    )

    out_features, reconstructed_image, input_image, z_mean, z_var = bigboi(input)
    loss_mse_v = loss_mse(target, out_features)
    loss_vae_v = loss_vae(reconstructed_image, input_image, z_mean, z_var)
    loss = loss_mse_v + loss_vae_v
    loss.backward()

    print('loss: {}'.format(loss.item()))

    print("features shape: {}".format(out_features.shape))
    print("reconstructed shape: {}".format(reconstructed_image.shape))
    print("z_mean: {} and z_var shapes: {}".format(z_mean.shape, z_var.shape))
