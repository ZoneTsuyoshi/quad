import torch
import torch.nn as nn


def construct_cnn_networks(n_cnn_layers, n_channels, cnn_channels, kernel_size, stride, padding, dilation, cnn_dropout, use_batchnorm, activation_function):
    if n_cnn_layers > 0:
        cnn_channels, kernel_size, stride, padding, dilation, cnn_dropout = map(lambda x: [x] * n_cnn_layers if isinstance(x, int) else x, [cnn_channels, kernel_size, stride, padding, dilation, cnn_dropout])
        conv = []

        for i in range(n_cnn_layers):
            in_channels = n_channels if i == 0 else cnn_channels[i - 1]
            out_channels = cnn_channels[i]
            conv.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[i], stride=stride[i], padding=padding[i], dilation=dilation[i]))
            if use_batchnorm:
                conv.append(nn.BatchNorm1d(out_channels))
            conv.append(getattr(nn, activation_function))
            if cnn_dropout[i] > 0:
                conv.append(nn.Dropout(cnn_dropout[i]))

        conv = nn.Sequential(*conv)
    else:
        conv = nn.Identity()
    return conv


def compute_output_length(input_length, n_cnn_layers, kernel_size, stride, padding, dilation):
    if n_cnn_layers == 0:
        return input_length
    cnn_channels, kernel_size, stride, padding, dilation = map(lambda x: [x] * n_cnn_layers if isinstance(x, int) else x, [cnn_channels, kernel_size, stride, padding, dilation])
    current_length = input_length
    for i in range(n_cnn_layers):
        current_length = (current_length + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
    return current_length


class LastIdentity(nn.Module):
    def __init__(self):
        super(LastIdentity, self).__init__()

    def forward(self, x):
        return x[:, -1]
    

class SequenceFlatten(nn.Module):
    def __init__(self):
        super(SequenceFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)