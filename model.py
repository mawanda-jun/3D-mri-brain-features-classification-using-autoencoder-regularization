# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
# import keras.backend as K
# from keras.losses import mse
# from keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense
# from keras.layers import Input, Reshape, Flatten, Dropout, SpatialDropout3D
# from keras.optimizers import adam
# from keras.models import Model
from group_norm import GroupNormalization


class GreenBlock(nn.Module):
    def __init__(self, filters):
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
        super().__init__()
        num_features = 53  # Number not valid # TODO: decide how big is num_features
        self.res = nn.Conv3d(num_features, out_channels=filters, kernel_size=(1, 1, 1), stride=1)
        self.block = nn.Sequential(OrderedDict([
            ('group_norm0', GroupNormalization(num_features, groups=8, padding=0)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv0', nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding='same')),  # TODO: define padding size for "same"
            ('group_norm1', GroupNormalization(filters, groups=8)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=(3, 3, 3), stride=1, padding='same')),  # TODO: define padding size for "same"
        ]))

    def forward(self, inputs):
        x_res = self.res(inputs)
        x = self.block(inputs)
        return torch.cat([x, x_res], dim=1)


# def green_block(inp, filters, name=None):
    # inp_res = nn.Conv3d(
    #     in_channels=None,  # TODO: define in channels here
    #     out_channels=filters,
    #     kernel_size=(1, 1, 1),
    #     stride=1)(inp)

    # axis=1 for channels_first data format
    # No. of groups = 8, as given in the paper
    # x = GroupNormalization(
    #     groups=8,
    #     axis=1)(inp)
    # x = F.relu(x)
    # x = nn.Conv3d(
    #     in_channels=None,  # TODO: define in channels here
    #     out_channels=filters,
    #     kernel_size=(3, 3, 3),
    #     stride=1,
    #     padding='same')(x)  # TODO: define same padding here
    #
    # x = GroupNormalization(
    #     groups=8,
    #     axis=1)(x)
    # # x = F.relu(x)
    # x = nn.Conv3d(
    #     in_channels=None,  # TODO: define in channels here
    #     out_channels=filters,
    #     kernel_size=(3, 3, 3),
    #     strides=1,
    #     padding='same')(x)  # TODO: define same padding here

    # out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    # out = torch.cat([x, inp_res], dim=0)  # TODO: understand which dimension must be concatenated
    # return out


# From keras-team/keras/blob/master/examples/variational_autoencoder.py

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_var = args
    batch = z_mean.shape[0]
    dim = z_mean.shape[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = torch.empty((batch, dim)).normal_()
    return z_mean + torch.exp(0.5 * z_var) * epsilon


class GTLoss(nn.Module):
    """
    loss_gt(e=1e-8)
    ------------------------------------------------------
    This function
    only calculates - L<dice> term of the following equation. (i.e. GT Decoder part loss)

    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>

    Parameters
    ----------
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.

    Returns
    -------
    loss_gt_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss.

    """
    def __init__(self, e):
        super().__init__()
        self.e = e

    def dice_coefficient(self, y_true, y_pred):
        intersection = torch.sum(torch.abs(y_true * y_pred), dim=[-3, -2, -1])
        dn = torch.sum(torch.square(y_true) + torch.square(y_pred), dim=[-3, -2, -1]) + self.e
        return torch.mean(2 * intersection / dn, dim=[0, 1])

    def forward(self, target, output):
        return 1 - self.dice_coefficient(target, output)


class VAELoss(nn.Module):
    def __init__(self, input_shape, weight_L2=0.1, weight_KL=0.1):
        super().__init__()
        """
        loss_VAE(input_shape, z_mean, z_var, weight_L2=0.1, weight_KL=0.1)
        ------------------------------------------------------
        Since keras does not allow custom loss functions to have arguments
        other than the true and predicted labels, this function acts as a wrapper
        that allows us to implement the custom loss used in the paper. This function
        calculates the following equation, except for -L<dice> term. (i.e. VAE decoder part loss)
        
        L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
        
        Parameters
        ----------
         `input_shape`: A 4-tuple, required
            The shape of an image as the tuple (c, H, W, D), where c is
            the no. of channels; H, W and D is the height, width and depth of the
            input image, respectively.
        `z_mean`: An keras.layers.Layer instance, required
            The vector representing values of mean for the learned distribution
            in the VAE part. Used internally.
        `z_var`: An keras.layers.Layer instance, required
            The vector representing values of variance for the learned distribution
            in the VAE part. Used internally.
        `weight_L2`: A real number, optional
            The weight to be given to the L2 loss term in the loss function. Adjust to get best
            results for your task. Defaults to 0.1.
        `weight_KL`: A real number, optional
            The weight to be given to the KL loss term in the loss function. Adjust to get best
            results for your task. Defaults to 0.1.
            
        Returns
        -------
        loss_VAE_(y_true, y_pred): A custom keras loss function
            This function takes as input the predicted and ground labels, uses them
            to calculate the L2 and KL loss.
            
        """
        self.input_shape = input_shape
        self.weight_KL = weight_KL
        self.weight_L2 = weight_L2

    def loss_VAE(self, y_true, y_pred, z_var, z_mean):
        c, H, W, D = self.input_shape
        n = c * H * W * D

        loss_L2 = torch.mean(torch.square(y_true - y_pred), dim=(1, 2, 3, 4))  # original axis value is (1,2,3,4).

        loss_KL = (1 / n) * torch.sum(torch.exp(z_var) + torch.square(z_mean) - 1. - z_var, dim=-1)

        return self.weight_L2 * loss_L2 + self.weight_KL * loss_KL

    def forward(self, target, output, z_var, z_mean):
        return self.loss_VAE(target, output, z_var, z_mean)


def build_model(input_shape=(4, 160, 192, 128), output_channels=3, weight_L2=0.1, weight_KL=0.1, dice_e=1e-8):
    """
    build_model(input_shape=(4, 160, 192, 128), output_channels=3, weight_L2=0.1, weight_KL=0.1)
    -------------------------------------------
    Creates the model used in the BRATS2018 winning solution
    by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)

    Parameters
    ----------
    `input_shape`: A 4-tuple, optional.
        Shape of the input image. Must be a 4D image of shape (c, H, W, D),
        where, each of H, W and D are divisible by 2^4, and c is divisible by 4.
        Defaults to the crop size used in the paper, i.e., (4, 160, 192, 128).
    `output_channels`: An integer, optional.
        The no. of channels in the output. Defaults to 3 (BraTS 2018 format).
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `dice_e`: Float, optional
        A small epsilon term to add in the denominator of dice loss to avoid dividing by
        zero and possible gradient explosion. This argument will be passed to loss_gt function.


    Returns
    -------
    `model`: A keras.models.Model instance
        The created model.
    """
    c, H, W, D = input_shape
    assert len(input_shape) == 4, "Input shape must be a 4-tuple"
    assert (c % 4) == 0, "The no. of channels must be divisible by 4"
    assert (H % 16) == 0 and (W % 16) == 0 and (D % 16) == 0, \
        "All the input dimensions must be divisible by 16"

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    ## Input Layer
    inp = Input(input_shape)

    ## The Initial Block
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_first',
        name='Input_x1')(inp)

    ## Dropout (0.2)
    x = SpatialDropout3D(0.2, data_format='channels_first')(x)

    ## Green Block x1 (output filters = 32)
    x1 = green_block(x, 32, name='x1')
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_32')(x1)

    ## Green Block x2 (output filters = 64)
    x = green_block(x, 64, name='Enc_64_1')
    x2 = green_block(x, 64, name='x2')
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_64')(x2)

    ## Green Blocks x2 (output filters = 128)
    x = green_block(x, 128, name='Enc_128_1')
    x3 = green_block(x, 128, name='x3')
    x = Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_128')(x3)

    ## Green Blocks x4 (output filters = 256)
    x = green_block(x, 256, name='Enc_256_1')
    x = green_block(x, 256, name='Enc_256_2')
    x = green_block(x, 256, name='Enc_256_3')
    x4 = green_block(x, 256, name='x4')

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    ## GT (Groud Truth) Part
    # -------------------------------------------------------------------------

    ### Green Block x1 (output filters=128)
    x = Conv3D(
        filters=128,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_128')(x4)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_128')(x)
    x = Add(name='Input_Dec_GT_128')([x, x3])
    x = green_block(x, 128, name='Dec_GT_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_64')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_64')(x)
    x = Add(name='Input_Dec_GT_64')([x, x2])
    x = green_block(x, 64, name='Dec_GT_64')

    ### Green Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_32')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_32')(x)
    x = Add(name='Input_Dec_GT_32')([x, x1])
    x = green_block(x, 32, name='Dec_GT_32')

    ### Blue Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_first',
        name='Input_Dec_GT_Output')(x)

    ### Output Block
    out_GT = Conv3D(
        filters=output_channels,  # No. of tumor classes is 3
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        activation='sigmoid',
        name='Dec_GT_Output')(x)

    ## VAE (Variational Auto Encoder) Part
    # -------------------------------------------------------------------------

    ### VD Block (Reducing dimensionality of the data)
    x = GroupNormalization(groups=8, axis=1, name='Dec_VAE_VD_GN')(x4)
    x = Activation('relu', name='Dec_VAE_VD_relu')(x)
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Dec_VAE_VD_Conv3D')(x)

    # Not mentioned in the paper, but the author used a Flattening layer here.
    x = Flatten(name='Dec_VAE_VD_Flatten')(x)
    x = Dense(256, name='Dec_VAE_VD_Dense')(x)

    ### VDraw Block (Sampling)
    z_mean = Dense(128, name='Dec_VAE_VDraw_Mean')(x)
    z_var = Dense(128, name='Dec_VAE_VDraw_Var')(x)
    x = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, z_var])

    ### VU Block (Upsizing back to a depth of 256)
    x = Dense((c // 4) * (H // 16) * (W // 16) * (D // 16))(x)
    x = Activation('relu')(x)
    x = Reshape(((c // 4), (H // 16), (W // 16), (D // 16)))(x)
    x = Conv3D(
        filters=256,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_VAE_ReduceDepth_256')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_VAE_UpSample_256')(x)

    ### Green Block x1 (output filters=128)
    x = Conv3D(
        filters=128,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_VAE_ReduceDepth_128')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_VAE_UpSample_128')(x)
    x = green_block(x, 128, name='Dec_VAE_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_VAE_ReduceDepth_64')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_VAE_UpSample_64')(x)
    x = green_block(x, 64, name='Dec_VAE_64')

    ### Green Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_VAE_ReduceDepth_32')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_VAE_UpSample_32')(x)
    x = green_block(x, 32, name='Dec_VAE_32')

    ### Blue Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_first',
        name='Input_Dec_VAE_Output')(x)

    ### Output Block
    out_VAE = Conv3D(
        filters=4,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_first',
        name='Dec_VAE_Output')(x)

    # Build and Compile the model
    out = out_GT
    model = Model(inp, outputs=[out, out_VAE])  # Create the model
    model.compile(
        adam(lr=1e-4),
        [loss_gt(dice_e), loss_VAE(input_shape, z_mean, z_var, weight_L2=weight_L2, weight_KL=weight_KL)],
        metrics=[dice_coefficient]
    )

    return model
