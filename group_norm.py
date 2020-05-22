import torch
import torch.nn.init as initializers


class GroupNormalization(torch.nn.Module):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        num_features: Integer, the number of incoming features (channels) to be normalized
        groups: Integer, the number of groups for Group Normalization.
        channel_first: Bool, tells the group the axis to be normalized is the first
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 num_features,
                 groups=32,
                 epsilon=1e-5,
                 channel_first=True,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 **kwargs):
        super().__init__()
        self.supports_masking = True
        self.groups = groups
        self.axis = 1 if channel_first else -1
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

        dim = num_features

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        shape = (dim,)

        if self.scale:
            self.beta = torch.nn.Parameter(torch.empty(shape), requires_grad=True)
            if self.beta_initializer == 'zeros':
                self.beta = initializers.zeros_(self.beta)
            else:
                raise ValueError("Bad beta_initializer type: choose 'zeros' or...")
        else:
            self.beta = None
        if self.center:
            self.gamma = torch.nn.Parameter(torch.empty(shape), requires_grad=True)
            if self.gamma_initializer == 'ones':
                self.gamma = initializers.ones_(self.gamma)
            else:
                raise ValueError("Bad gamma_initializer type: choose 'zeros' or...")
        else:
            self.gamma = None

        # self.built = True

    def forward(self, inputs, **kwargs):
        input_shape = inputs.shape

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        reduction_axes = reduction_axes[0:self.axis] + reduction_axes[self.axis + 1:]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape = broadcast_shape[0:1] + [self.groups] + broadcast_shape[1:]  # Prepare broadcast shape in position 1

        reshape_group_shape = inputs.shape
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes = group_axes[0:1] + [self.groups] + group_axes[1:]
        # group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = torch.tensor(group_shape)
        inputs = torch.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = torch.mean(inputs, dim=group_reduction_axes, keepdim=True)
        variance = torch.var(inputs, dim=group_reduction_axes, keepdim=True)

        inputs = (inputs - mean) / (torch.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = torch.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = torch.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = torch.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = torch.reshape(outputs, input_shape)

        return outputs

    #
    # def compute_output_shape(self, input_shape):
    #     return input_shape


if __name__ == '__main__':
    ip = torch.rand(128, 4, 48, 48, 48)  # (batch, c, H, W, D)
    #ip = Input(batch_shape=(100, None, None, 2))
    x = GroupNormalization(num_features=4, groups=2, epsilon=0.1)(ip)
    print(x.shape)


