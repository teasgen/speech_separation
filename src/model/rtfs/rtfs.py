# contains simple dprnn and attention

import torch
import torch.nn as nn
# from sru import SRU # TODO: remove when cuda is accessible


# dprnn with sru from https://arxiv.org/pdf/2309.17189
class DPRNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 4,
        hidden_size: int = 32,
        kernel_size: int = 8,
        size: int = 4
    ):
        super(DPRNN, self).__init__()

        self.transpose = (size == 4)
        # to avoid unnecessary reshaping
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=in_channels
        )

        # TODO: check for sizes
        self.unfold = nn.Unfold(
            kernel_size=(kernel_size, 1),
            stride=(1, 1)
        )

        self.sru = SRU(
            input_size=in_channels * kernel_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=hidden_size * 2,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1
        )

    def forward(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(-1, -2).contiguous()
            # TODO: check transpose

        res = x
        B, C, T, F = x.shape

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * F, C, T, 1)
        # [B*F, C, T, 1]

        x = self.unfold(x)
        # [B*F, C*8, *]
        x = x.permute(2, 0, 1)
        # [*, B*F, C*8]

        x = self.sru(x)
        x = x.permute(1, 2, 0)
        # [B*F, C*8, *]

        x = self.conv_transpose(x)
        x = x.view([B, F, C, T])
        x.permute(0, 2, 3, 1).contiguous()
        # [B, C, T, F]

        x = x + res
        # TODO: maybe cut?
        if self.transpose:
            x = x.transpose(-1, -2).contiguous()

        return x


# LN basic block from https://arxiv.org/pdf/2209.03952
class LNBlock(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super(LNBlock, self).__init__()
        # [out_channels, num_freqs]
        param_size = [1, dims[0], 1, dims[1]]
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps
        self.dim = (1, 3)  # norm by C and F

    def forward(self, x):
        # x: [B, C, T, F]
        mu = x.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(x.var(dim=self.dim, keepdim=True, unbiased=False) + self.eps)
        x_hat = (x - mu) / std
        return x_hat * self.gamma + self.beta


# basic block in https://arxiv.org/pdf/2209.03952
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.act = nn.PReLU()
        self.norm = LNBlock((out_channels, 64))

    def forward(self, x):
        # x: [B, in_channels, T, F]
        x = self.conv(x) # [B, out_channels, T, F]
        x = self.act(x)
        x = self.norm(x)
        return x


# link: https://arxiv.org/pdf/2209.03952
class TFDecepticon(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            out_channels: int = 64,
            num_heads: int = 4,
            num_freqs: int = 64,
            hidden_size: int = 4
    ):
        super(TFDecepticon, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size # hidden size for Q_l and K_l
        self.num_freqs = num_freqs
        self.num_heads = num_heads
        self.embed_size = num_heads * num_freqs

        assert self.in_channels % self.num_heads == 0
        assert self.in_channels == self.out_channels

        self.q = nn.ModuleList([ConvBlock(self.in_channels, self.hidden_size) for _ in range(self.num_heads)])
        self.k = nn.ModuleList([ConvBlock(self.in_channels, self.hidden_size) for _ in range(self.num_heads)])
        self.v = nn.ModuleList([ConvBlock(self.in_channels, self.in_channels // self.num_heads) for _ in range(self.num_heads)])
        self.softmax = nn.Softmax(dim=-1)

        self.likear = ConvBlock(self.in_channels, self.in_channels)

    def forward(self, x):
        # x: [B, C, T, F], C = 4
        res = x
        queries = [q(x) for q in self.q] # [B, 4, T, F]
        keys = [k(x) for k in self.k]
        values = [v(x) for v in self.v] #[B, 1, T, F]

        Q_l = torch.cat(queries, dim=0)  # [4*B, 4, T, F]
        K_l = torch.cat(keys, dim=0)
        V_l = torch.cat(values, dim=0)  # [4*B, 1, T, F]

        Q_l = Q_l.transpose(1, 2).flatten(start_dim=2)  # [4*B, T, 4*F]
        K_l = K_l.transpose(1, 2).flatten(start_dim=2)

        V_l = V_l.transpose(1, 2) # [4*B, T, 1, F]
        old_shape = V_l.shape
        V_l = V_l.flatten(start_dim=2) # [4*B, T, F]                 

        attention_matrix = self.softmax(
            torch.matmul(Q_l, K_l.transpose(1, 2)) / torch.sqrt(self.embed_size)
        )
        # [4*B, T, T]

        # attn
        A_l = torch.matmul(attention_matrix, V_l) # [4*B, T, F]
        A_l = A_l.view(old_shape) # [4*B, T, 1, F]
        A_l = A_l.transpose(1, 2) # [4*B, 1, T, F]

        B, C, T, F = x.shape
        x = A_l.view(self.num_heads, B, self.in_channels // self.num_heads, T, F)  # [4, B, 1, T, F]
        x = x.transpose(0, 1).contiguous() # [B, 4, 1, T, F]
        x = x.view(B, self.in_channels, T, F) # [B, 4, T, F]
        x = self.likear(x) # [B, 4, T, F]
        x = x + res # [B, 4, T, F]

        return x
    

class RTFSBlock(nn.Module):
    def __init__(
        self
    ):
        super(RTFSBlock, self).__init__()

        freq_dprnn = DPRNN(
            in_channels=64,
            num_layers=4,
            hidden_size=32,
            kernel_size=8,
            size=4
        )
        time_dprnn = DPRNN(
            in_channels=64,
            num_layers=4,
            hidden_size=32,
            kernel_size=8,
            size=3
        )
        tf_attention = TFDecepticon(
            in_channels=64,
            out_channels=64,
            num_heads=4,
            num_freqs=64,
            hidden_size=4
        )

        self.net = nn.Sequential(
            freq_dprnn,
            time_dprnn,
            tf_attention
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
        