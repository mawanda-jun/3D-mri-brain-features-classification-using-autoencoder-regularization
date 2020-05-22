def calc_same_padding(i, k, s):
    """
    Calculate same padding for convolutions
    :param i: input size (simpler for squared images)
    :param k: kernel_size
    :param s: stride
    :return: padding for 'same' padding
    """
    return round(((i - 1) * s - i + k) / 2 + 0.1)


def calc_conv_shape(i, k, p, s):
    return round((i - k + 2 * p) / s + 0.1) + 1


def calc_deconv_shape(i, k, p, s, out_p):
    return (i -1)*s - 2*p + k + out_p

