import torch
import torch.nn as nn
from utils import *
# from group_norm import GroupNormalization
from collections import OrderedDict

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_side_dim):
        """
        green_block(inp, filters, name=None)
        ------------------------------------
        Implementation of the special residual block used in the paper. The block
        consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
        units, with a residual connection from the input `inp` to the output. Used
        internally in the model. Can be used independently as well.

        Note that images must come with dimensions "c, H, W, D"

        Parameters
        ----------
        `inp`: An keras.layers.layer instance, required
            The keras layer just preceding the green block.
        `out_channels`: integer, required
            No. of filters to use in the 3D convolutional block. The output
            layer of this green block will have this many no. of channels.

        Returns
        -------
        `out`: A keras.layers.Layer instance
            The output of the green block. Has no. of channels equal to `filters`.
            The size of the rest of the dimensions remains same as in `inp`.
        """
        super(GreenBlock, self).__init__()
        out_channels //= 2
        self.res = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        # Calculate output shape to predict padding dimension
        out_dim = calc_conv_shape(input_side_dim, 1, 0, 1)
        next_padding = calc_same_padding(out_dim, 3, 1)
        # Define block
        self.block = nn.Sequential(OrderedDict([
            # ('group_norm0', GroupNormalization(in_channels, groups=8, padding=0)),
            ('norm0', nn.BatchNorm3d(num_features=in_channels)),
            ('relu0', nn.LeakyReLU(inplace=True)),
            ('conv0', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=next_padding)),
            # Since padding is "same" we can keep the same padding value for the next convolution
            # ('group_norm1', GroupNormalization(out_channels, groups=8)),
            ('norm1', nn.BatchNorm3d(num_features=out_channels)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=next_padding)),
        ]))

    def forward(self, inputs):
        x_res = self.res(inputs)
        x = torch.nn.functional.dropout(self.block(inputs), p=0.4, training=self.training)
        return torch.cat([x, x_res], dim=1)


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
class Reparametrization(nn.Module):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

    def __init__(self):
        super(Reparametrization, self).__init__()

    def forward(self, z_mean, z_var):
        if self.training:
            std = torch.exp(0.5 * z_var)
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = torch.empty_like(z_var, device=z_mean.device).normal_()
            return z_mean + std*epsilon
        else:
            return z_mean


class UpGreenBlock(nn.Sequential):
    def __init__(self, in_features, out_features, input_side_dim):
        super(UpGreenBlock, self).__init__()

        self.add_module('conv', nn.Conv3d(in_features, out_features, kernel_size=1, stride=1))
        # Calculate output shape to predict padding dimension
        out_dim = calc_conv_shape(input_side_dim, 1, 0, 1)
        # Since scale_factor == 2:
        side_dim = out_dim*2
        self.add_module('up', nn.Upsample(scale_factor=2))
        self.add_module('green', GreenBlock(out_features, out_features, side_dim))