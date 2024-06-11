import torch
import torch.nn as nn
from einops import rearrange

def get_attention_layers(attn_type, input_dim, hidden_dim, n_heads, dropout):
    if attn_type == 'raw':
        return RawAttention(dropout)
    elif attn_type == 'simple':
        return SimpleAttention(input_dim, hidden_dim, dropout)
    elif attn_type == 'multihead':
        return MultiHeadAttention(input_dim, hidden_dim, n_heads, dropout)
    else:
        raise ValueError('Invalid attention type')


class RawAttention(nn.Module):
    def __init__(self,
                 dropout: float = 0.):
        super(RawAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_channels]
        """
        q = k = v = x
        attn = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5) # [batch_size, seq_len, seq_len]
        attn = torch.softmax(attn, dim=-1) # [batch_size, seq_len, seq_len]
        attn = self.dropout(attn)
        x = torch.matmul(attn, v) # [batch_size, seq_len, n_channels]
        return x, attn


class SimpleAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.):
        super(SimpleAttention, self).__init__()
        self.to_qkv = nn.Linear(input_dim, 3*hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_channels]
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # [batch_size, seq_len, hidden_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5) # [batch_size, seq_len, seq_len]
        attn = torch.softmax(attn, dim=-1) # [batch_size, seq_len, seq_len]
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        return x, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 dropout: float = 0.):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.to_qkv = nn.Linear(input_dim, 3*hidden_dim*n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim*n_heads, input_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_channels]
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1) # 3 x [batch_size, seq_len, n_heads*hidden_dim]
        q, k, v = map(lambda t: rearrange(t, 'b s (n h) -> b s n h', n=self.n_heads, h=self.hidden_dim), qkv) # [batch_size, seq_len, n_heads, hidden_dim]
        attn = torch.einsum('b s n h, b t n h -> b n s t', q, k) / (k.size(-1) ** 0.5) # [batch_size, n_heads, seq_len, seq_len]
        attn = torch.softmax(attn, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
        attn = self.dropout(attn)
        x = torch.einsum('b n s t, b t n h -> b s n h', attn, v) # [batch_size, seq_len, n_heads, hidden_dim]
        x = rearrange(x, 'b s n h -> b s (n h)') # [batch_size, seq_len, n_heads*hidden_dim]
        x = self.out(x) # [batch_size, seq_len, n_channels]
        return x, attn