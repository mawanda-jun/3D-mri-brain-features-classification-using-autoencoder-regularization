import torch
import torch.nn as nn

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
    def __init__(self, weight_L2=0.1, weight_KL=0.1):
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
        # c, H, W, D = input_shape
        # self.N = c * H * W * D
        self.weight_KL = weight_KL
        self.weight_L2 = weight_L2

    def loss(self, y_pred, y_true, z_mean, z_var):
        # loss_L2 = torch.mean(torch.square(y_true - y_pred), dim=(1, 2, 3, 4))  # original axis value is (1,2,3,4).
        loss_L2 = torch.nn.functional.mse_loss(y_pred, y_true)  # original axis value is (1,2,3,4).

        # loss_KL = (1 / self.N) * torch.sum(torch.exp(z_var) + torch.square(z_mean) - 1. - z_var, dim=-1)
        loss_KL = 0.5 * torch.sum(z_var.exp() + z_mean.pow(2) - 1. - z_var) / y_pred.size(0)

        return self.weight_L2 * loss_L2 + self.weight_KL * loss_KL

    def forward(self, reconstructed_image, input_image, z_mean, z_var):
        return self.loss(reconstructed_image, input_image, z_mean, z_var)