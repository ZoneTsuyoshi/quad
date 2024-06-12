from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .attention import get_attention_layers

class QuADNet(nn.Module):
    def __init__(self,
                 n_channels: int,
                 horizon: int = 10,
                 window_size: int = 60,
                 rnn_hidden: int = 16,
                 cnn_channels: Union[int, List[int], Tuple[int]] = 16,
                 attn_hidden: Union[int, List[int], Tuple[int]] = 16,
                 n_rnn_layers: int = 1,
                 n_cnn_layers: int = 1,
                 n_output_layers: int = 1,
                 activation_function: str = 'SiLU',
                 kernel_size: Union[int, List[int], Tuple[int]] = 3,
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 use_batchnorm: bool = False,
                 cnn_dropout: float = 0.,
                 rnn_type: str = 'GRU',
                 rnn_dropout: float = 0.,
                 bidirectional: bool = False,
                 attn_type: str = 'simple',
                 n_heads: int = 1,
                 attn_dropout: Union[float, List[float], Tuple[float]] = 0.,
                 use_last_stat: bool = False,
                 use_reconstruction: bool = False):
        super(QuADNet, self).__init__()

        self.dim = n_channels
        self.horizon = horizon
        self.window_size = window_size

        # construct CNN layers
        self.cnn = construct_cnn_networks(n_cnn_layers, n_channels, cnn_channels, kernel_size, stride, padding, dilation, cnn_dropout, use_batchnorm, activation_function)
        cnn_output_channels = cnn_channels if isinstance(cnn_channels, int) else cnn_channels[-1]
        cnn_output_length = compute_output_length(window_size, n_cnn_layers, kernel_size, stride, padding, dilation)

        # construct RNN layers
        self.use_rnn = n_rnn_layers > 0
        if self.use_rnn:
            rnn = getattr(nn, rnn_type)
            self.rnn = rnn(input_size=cnn_output_channels, hidden_size=rnn_hidden, num_layers=n_rnn_layers, dropout=rnn_dropout, bidirectional=bool(bidirectional), batch_first=True)
            rnn_output_channels = rnn_hidden * (1 + bidirectional)
        else:
            self.rnn = nn.Linear(cnn_output_channels, rnn_hidden)
            rnn_output_channels = rnn_hidden
        self.rnn_act = getattr(nn, activation_function)()

        # construct attention layers
        self.attn = get_attention_layers(attn_type, rnn_output_channels, attn_hidden, n_heads, attn_dropout)

        # reconstruction layers
        self.use_reconstruction = use_reconstruction
        self.recon = nn.Linear(rnn_output_channels, n_channels) if use_reconstruction else None

        # construct stat layers
        if use_last_stat:
            self.stat = LastIdentity()
            stat_output_channels = rnn_output_channels
        else:
            self.stat = SequenceFlatten()
            stat_output_channels = rnn_output_channels * cnn_output_length

        # construct output layers
        if n_output_layers == 1:
            self.output = nn.Linear(stat_output_channels, horizon*n_channels)
        else:
            self.output = []
            for _ in range(n_output_layers-1):
                self.output.append(nn.Linear(stat_output_channels, stat_output_channels))
                self.output.append(getattr(nn, activation_function))
            self.output.append(nn.Linear(stat_output_channels, horizon*n_channels))
            self.output = nn.Sequential(*self.output)


    def forward(self, x, return_attn: bool = False):
        """
        Args:
            x: [batch_size, seq_len, n_channels]
        """
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2) # [batch_size, seq_len, cnn_output_channels]
        if self.use_rnn:
            h0 = torch.zeros(self.rnn.num_layers*(1+self.rnn.bidirectional), x.size(0), self.rnn.hidden_size).to(x.device)
            x, _ = self.rnn(x, h0) # [batch_size, seq_len, rnn_output_channels]
        else:
            x = self.rnn(x) # [batch_size, seq_len, rnn_output_channels]
        x = self.rnn_act(x) # [batch_size, seq_len, rnn_output_channels]
        x, attn = self.attn(x) # [batch_size, seq_len, rnn_output_channels], [batch_size, seq_len, seq_len]
        if self.use_reconstruction:
            r = self.recon(x) # [batch_size, seq_len, n_channels]
        x = self.stat(x) # [batch_size, stat_output_channels]
        x = self.output(x) # [batch_size, horizon*n_channels]
        x = x.view(x.size(0), self.horizon, self.dim) # [batch_size, horizon, n_channels]
        if self.use_reconstruction:
            outputs = [x, r]
        else:
            outputs = x
        if return_attn:
            return outputs, attn
        return outputs
