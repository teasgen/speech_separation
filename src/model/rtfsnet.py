import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_avg_pool1d, interpolate



class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int=2,
        out_channels: int = 256,
    ):
        super(AudioEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels),
            # nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
        )

    def forward(self, complex_spectrogram: torch.Tensor):
        # [B, C_in, T, F]   
        encoded = self.net(complex_spectrogram) # [B, C_out, T, F]

        return encoded


# dprnn basic block with lstm instead of sru
class DPRNNUnit(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 4,
        hidden_size: int = 32,
        kernel_size: int = 8,
        transpose: bool = True
    ):
        super(DPRNNUnit, self).__init__()
        self.norm = ChanNorm(dims=(in_channels, 1))
        self.unfold = nn.Unfold(kernel_size=(8, 1), stride=(1, 2))
        self.kernel_size = kernel_size
        self.lstm = nn.LSTM(
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
        self.transpose = transpose

    def forward(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(-1, -2).contiguous()

        res = x
        B, C, T, F = x.shape
        # nominal notation, in fact at first forward its [B, C, F, T]
        # cause we transpose to process frequency

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * F, C, T, 1)
        # [B*F, C, T, 1]
        x = self.unfold(x)
        # [B*F, C*8, L]
        # x = x.unfold(dimension=2, size=self.kernel_size, step=1)  # [B*F, C, L, kernel_size]
        x = x.permute(2, 0, 1)  # [L, B*F, C, kernel_size]
        # x = x.reshape(-1, B * F, C * self.kernel_size)  # [L, B*F, C * kernel_size]
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)  # [B*F, hidden_size*2, L] = [B*F, 64, 55] or [B*F, 64, 116]

        x = self.conv_transpose(x)
        x = x.view(B, F, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, C, T, F]

        x = x + res
        if self.transpose:
            x = x.transpose(-1, -2).contiguous()

        return x
    

# normalize by channels
class ChanNorm(nn.Module):
    def __init__(
        self,
        dims=(64, 1),
        eps=1e-5,
        affine=True,
    ):
        super(ChanNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.norm_dims = (1,) #nly channels

        if self.affine:
            dims = [1, dims[0], 1, dims[1]]
            self.weight = nn.Parameter(torch.ones(*dims))
            self.bias = nn.Parameter(torch.zeros(*dims))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mu = x.mean(dim=self.norm_dims, keepdim=True)
        std = x.std(dim=self.norm_dims, keepdim=True, unbiased=False)
        x_hat = (x - mu) / (std + self.eps)
        if self.affine:
            return x_hat * self.weight + self.bias
        return x_hat
            


# cfLN in https://arxiv.org/pdf/2209.03952
class ChanFreqNorm(nn.Module):
    def __init__(
        self,
        dims=(4, 64),
        eps=1e-5,
        affine=True,
    ):
        super(ChanFreqNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.norm_dims = (1, 3)  # normalize along C and F
        
        if self.affine:
            dims = [1, dims[0], 1, dims[1]]
            self.weight = nn.Parameter(torch.ones(*dims))
            self.bias = nn.Parameter(torch.zeros(*dims))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mu = x.mean(dim=self.norm_dims, keepdim=True)
        std = x.std(dim=self.norm_dims, keepdim=True, unbiased=False)
        x_hat = (x - mu) / (std + self.eps)
        if self.affine:
            return x_hat * self.weight + self.bias
        return x_hat


# link: https://arxiv.org/pdf/2209.03952
class TFDecepticon(nn.Module):
    """Regular attention block, but based on convolutions,
    as described in https://arxiv.org/pdf/2209.03952 figure 2.
    """
    
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

        self.q = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1),
                nn.PReLU(),
                ChanFreqNorm()
            )
        for _ in range(num_heads)])
        self.k = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1),
                nn.PReLU(),
                ChanFreqNorm((hidden_size, num_freqs))
            )
        for _ in range(num_heads)])
        self.v = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels // num_heads, kernel_size=1),
                nn.PReLU(),
                ChanFreqNorm((in_channels // num_heads, num_freqs))
            )
        for _ in range(num_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.PReLU(),
            ChanFreqNorm((in_channels, num_freqs))
        )

    def forward(self, x):
        # x: [B, C, T, F], C = 64, same as D in the paper
        res = x
        queries = [q(x) for q in self.q] # [B, 4, T, F]
        keys = [k(x) for k in self.k]
        values = [v(x) for v in self.v] #[B, 16, T, F]

        Q_l = torch.cat(queries, dim=0)  # [4*B, 4, T, F]
        K_l = torch.cat(keys, dim=0)
        V_l = torch.cat(values, dim=0)  # [4*B, 16, T, F]

        Q_l = Q_l.transpose(1, 2).flatten(start_dim=2)  # [4*B, T, 4*F]
        K_l = K_l.transpose(1, 2).flatten(start_dim=2)

        V_l = V_l.transpose(1, 2) # [4*B, T, 1, F]
        target_shape = V_l.shape
        V_l = V_l.flatten(start_dim=2) # [4*B, T, F]                 

        attention_matrix = self.softmax(
            torch.matmul(Q_l, K_l.transpose(1, 2)) / self.embed_size**0.5
        )
        # [4*B, T, T]

        # attn
        A_l = torch.matmul(attention_matrix, V_l) # [4*B, T, F]
        A_l = A_l.view(target_shape) # [4*B, T, 16, F]
        A_l = A_l.transpose(1, 2) # [4*B, 16, T, F]

        B, C, T, F = x.shape
        x = A_l.view(self.num_heads, B, self.in_channels // self.num_heads, T, F)  # [4, B, 1, T, F]
        x = x.transpose(0, 1).contiguous() # [B, 4, 16, T, F]
        x = x.view(B, self.in_channels, T, F) # [B, 4, T, F]
        x = self.linear(x) # [B, 4, T, F]
        x = x + res # [B, 4, T, F]

        return x


class DPRNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 8,
        hidden_size_rnn: int = 32,
        hidden_size_attn: int = 4,
        num_heads: int = 4,
        num_freqs: int = 64
    ):
        super(DPRNNBlock, self).__init__()

        freq_dprnn = DPRNNUnit(
            in_channels=in_channels,
            num_layers=num_layers,
            hidden_size=hidden_size_rnn,
            kernel_size=kernel_size,
            transpose=True
        )
        time_dprnn = DPRNNUnit(
            in_channels=in_channels,
            num_layers=num_layers,
            hidden_size=hidden_size_rnn,
            kernel_size=kernel_size,
            transpose=False
        )
        tf_attention = TFDecepticon(
            in_channels=in_channels,
            out_channels=in_channels,
            num_heads=num_heads,
            num_freqs=num_freqs,
            hidden_size=hidden_size_attn
        )

        self.net = nn.Sequential(
            freq_dprnn,
            time_dprnn,
            tf_attention
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# see reconstruction phase from https://arxiv.org/pdf/2309.17189
class TFARUnit(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 4
    ):
        super(TFARUnit, self).__init__()
        self.w3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels)
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )
        self.w1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.Sigmoid(),
        )
    

    def forward(self, m: torch.Tensor, n: torch.Tensor):
        """
        In paper's notation, this block is 
        I(m, n) = φ (sigmoid (W1 (n))) ⊙ W2 (m) + φ (W3 (n))
        (copied formula paper :)

        m is A_i, downsampled x
        n is A_G, attentioned features
        """
        B, C, T, F = m.shape
        term1 = self.w1(n)
        # TODO: maybe add smarter interpolation?
        term1 = interpolate(term1, size=(T, F), mode="nearest")
        
        # TODO: maybe here too
        term3 = self.w3(n)
        term3 = interpolate(term3, size=(T, F), mode="nearest")

        
        term2 = self.w2(m)
        term2 = interpolate(term2, size=(T, F), mode="nearest")
        upsampled = term1 * term2 + term3
        return upsampled


class RTFSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        hidden_size: int = 64,
        kernel_size: int = 4,
    ):        
        super(RTFSBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        self.skip_connection = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                groups=in_channels,
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=self.in_channels),
            nn.PReLU(),)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=1
            ),
            nn.GroupNorm(num_groups=1, num_channels=self.hidden_size),
            nn.PReLU(),)

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, groups=hidden_size, stride=1, padding=(0, 1)),
            nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, groups=hidden_size, stride=2, padding=(0, 1)),
            nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        )
        # not sure about output_size
        # self.pooling = nn.AdaptiveAvgPool2d(output_size=[audio_len//2, num_freqs//2])
        self.dual_path = DPRNNBlock(in_channels=self.hidden_size)
        self.reconstruction_1 = TFARUnit(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.reconstruction_2 = TFARUnit(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.upsample = TFARUnit(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.final_conv = nn.Conv2d(in_channels=hidden_size, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        B, C, T, F = x.shape
        res = self.skip_connection(x)

        x_cnn = self.cnn(x)
        x_compressed_1 = self.downsample_1(x_cnn)
        x_compressed_2 = self.downsample_2(x_compressed_1)

        target_shape = x_compressed_2.shape
        Ag_1 = adaptive_avg_pool2d(x_compressed_1, output_size=target_shape[-2:])
        Ag_2 = adaptive_avg_pool2d(x_compressed_2, output_size=target_shape[-2:])
        Ag = Ag_1 + Ag_2
        Ag = self.dual_path(Ag)
        # formulae (13) - (16)

        reconstructed_1 = self.reconstruction_1(x_compressed_1, Ag)
        reconstructed_2 = self.reconstruction_2(x_compressed_2, Ag)
        upsampled = self.upsample(reconstructed_1, reconstructed_2) + x_compressed_1
        # formulae (17) - (19)

        upsampled = interpolate(self.final_conv(upsampled), size=(T, F), mode="nearest")
        # to match sizes
        return upsampled + res        


class TFARUnitVideo(nn.Module):
    """TFAR unit but for video. Convolutions are 1D and using Batch Normalization.
    Yet the logic remains the same.
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3
    ):
        super(TFARUnitVideo, self).__init__()
        self.w3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.w2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.w1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=in_channels, bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, m: torch.Tensor, n: torch.Tensor):
        """
        In paper's notation, this block is 
        I(m, n) = φ (sigmoid (W1 (n))) ⊙ W2 (m) + φ (W3 (n))
        (copied formula paper :)

        m is A_i, downsampled x
        n is A_G, attentioned features
        """
        # TODO: check m and n sizes
        B, C, T = m.shape
        term1 = self.w1(n)
        # TODO: maybe add smarter interpolation?
        term1 = interpolate(term1, size=(T), mode="nearest")
        
        # TODO: maybe here too
        term3 = self.w3(n)
        term3 = interpolate(term3, size=(T), mode="nearest")

        
        term2 = self.w2(m)
        term2 = interpolate(term2, size=(T), mode="nearest")
        upsampled = term1 * term2 + term3
        return upsampled


# attention fusion block from https://arxiv.org/pdf/2309.17189
# formulae (5) - (8)
class CAF(nn.Module):
    """CAF block from paper. Includes attention and gated fusion.
    Notation kept close to the paper's notation.
    """
    def __init__(
        self,
        in_channels_attn: int,
        in_channels_gt: int,
        num_attn_heads: int = 1
    ):
        super(CAF, self).__init__()
        self.in_channels_attn = in_channels_attn
        self.in_channels_gt = in_channels_gt
        self.num_attn_heads = num_attn_heads

        self.P1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_attn, out_channels=in_channels_attn, kernel_size=1, groups=in_channels_attn, bias=False),
            nn.BatchNorm2d(num_features=in_channels_attn),
        )
        self.P2_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_attn, out_channels=in_channels_attn, kernel_size=1, groups=in_channels_attn, bias=False),
            nn.BatchNorm2d(num_features=in_channels_attn),
        )
        self.F1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels_gt, out_channels=num_attn_heads * in_channels_attn, kernel_size=1, groups=in_channels_attn),
            nn.GroupNorm(num_groups=1, num_channels=num_attn_heads * in_channels_attn)
        )
        self.F2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels_gt, out_channels=in_channels_attn, kernel_size=1, groups=in_channels_attn),
            nn.GroupNorm(num_groups=1, num_channels=in_channels_attn)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        pass

    
    def forward(self, a1: torch.Tensor, v1: torch.Tensor):
        B, C, T, F = a1.shape
        a_val = self.P1(a1)
        a_gate = self.P2_relu(a1)

        # ATTENTION FUSION:
        v_h = self.F1(v1)
        v_h = v_h.reshape(B, self.in_channels_attn, self.num_attn_heads, -1)
        v_mean = v_h.mean(2, keepdim=False)
        v_mean = v_mean.view(B, self.in_channels_attn, -1)
        # TODO; maybe make smarter interpolation?
        v_attn = interpolate(self.softmax(v_mean), size=T, mode="nearest")


        # GATED FUSION:
        v_key = interpolate(self.F2(v1), size=T, mode="nearest")
        v_key = v_key.unsqueeze(-1)
        f2 = a_gate * v_key

        v_attn = v_attn.unsqueeze(-1)
        f1 = v_attn * a_val

        # Fusion
        fusion = f1 + f2

        return fusion


# adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoder(nn.Module):
    def __init__(
        self, 
        max_length=10000, 
        embed_dim=64, 
        dropout=0.1,
        device='cuda:0' #default deivce
    ):
        super(PositionalEncoder, self).__init__()
        self.pos_features = torch.zeros(max_length, embed_dim, device=device)

        positions = torch.arange(0, max_length, dtype=torch.float, device=device).unsqueeze(1)
        freqs = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / embed_dim)
        )
        arguments = positions * freqs
        self.pos_features[:, 0::2] = torch.sin(arguments)
        self.pos_features[:, 1::2] = torch.cos(arguments)
        self.pos_features = self.pos_features.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pos_features[:, :x.shape[1]]
        return self.dropout(x)


# Global Attention module from https://arxiv.org/pdf/2209.15200
# see section 3.3 for this module
class GAModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        kernel_size: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        super(GAModule, self).__init__()

        # attention
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.positional_encoder = PositionalEncoder(embed_dim=in_channels, dropout=dropout_rate)
        self.attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # feed-forward
        fc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels*2)
        )
        extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=kernel_size, groups=in_channels*2),
            nn.GroupNorm(num_groups=1, num_channels=in_channels*2),
            nn.ReLU()
        )
        dropout_ffn = nn.Dropout(dropout_rate)
        fc2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels)
        )
        self.ffn = nn.Sequential(
            fc1, extractor, dropout_ffn, fc2
        )


    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2)

        x = self.norm1(x)
        x = self.positional_encoder(x)
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout1(x) + residual
        x = self.norm2(x)

        x = x.transpose(2, 1)
        x = x + res

        # TODO: add padding or interpolation
        res_ffn = x
        x = self.ffn(x)
        x = interpolate(x, size=res_ffn.shape[-1], mode='linear', align_corners=False)
        # to match sizes
        x = self.dropout2(x) + res_ffn
        return x


# tda from https://arxiv.org/pdf/2209.15200,
# adapted for rfts-net video processing architecture
class VideoProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        hidden_size: int = 64,
        kernel_size: int = 3,
        num_upsamples: int = 4,
        num_downsamples: int = 4
    ):
        super(VideoProcessor, self).__init__()
        self.num_upsamples = num_upsamples
        self.num_downsamples = num_downsamples
        self.skip_connection = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels),
            nn.BatchNorm1d(num_features=in_channels),
            nn.PReLU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.PReLU()
        )
        self.downsamples = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1 if i==0 else 2, groups=hidden_size),
                nn.BatchNorm1d(num_features=hidden_size),
            ) for i in range(num_downsamples)]
        )
        
        self.video_attn = GAModule(in_channels=hidden_size, kernel_size=kernel_size, num_heads=8, dropout_rate=0.1)

        self.reconstructions = nn.ModuleList(
            [TFARUnitVideo(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size) \
            for _ in range(num_upsamples)]
        )
        self.upsamples = nn.ModuleList(
            [TFARUnitVideo(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size) \
            for _ in range(num_upsamples - 1)]
        )
        self.final_conv = nn.Conv1d(in_channels=hidden_size, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B, 512, T] = [B, 512, 50]
        res = self.skip_connection(x)
        x_cnn = self.cnn(x)

        x_compressed = []
        current = x_cnn
        for downsample in self.downsamples:
            current = downsample(current)
            x_compressed.append(current)
        # compressed time dimension in 2**(num_downsamples-1) times

        B, C, T = x_compressed[-1].shape # approx 5
        A_g_list = [
            adaptive_avg_pool1d(compression, output_size=(T,))
            for compression in x_compressed
        ]
        A_g = sum(A_g_list)
        A_g = self.video_attn(A_g)
        # TODO: debug reconstructions
        reconstructions = [
            reconstruction(compression, A_g)
            for reconstruction, compression in zip(self.reconstructions, x_compressed)
        ]

        upsamples = reconstructions[-1]
        for upsample, reconstruction, downsample in zip(
            reversed(self.upsamples), reversed(reconstructions[:-1]), reversed(x_compressed[:-1])
        ):
            upsamples = upsample(reconstruction, upsamples) + downsample

        # TODO: maybe add smarter interpolation?
        upsamples = interpolate(upsamples, size=(x.shape[-1]), mode="nearest")

        return self.final_conv(upsamples) + res


class AudioProcessor(nn.Module):
    def __init__(
        self,
        R: int = 4,
        in_channels: int = 256,
        out_channels: int = 64,
        hidden_size: int = 64,
        kernel_size: int = 4,
    ):
        super(AudioProcessor, self).__init__()
        self.R = R
        self.net = RTFSBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor):
        res = x
        for i in range(self.R - 1):
            x = self.net((x) if i == 0 else (x + res))
        return x
        

class SeparationNetwork(nn.Module):
    """Full separation network in RTFS-Net architecture pipeline.
    Default hyperparameters are for the case of R=4 from the paper.
    The CAF block and VP and AP blocks are both present here because
    the authors use a shared architecture for AP and R=4 stacked RTFS blocks.
    The CAF is between them so it has to be present here too.
    """
    def __init__(
        self,
        R: int = 4,
        audio_channels: int = 256,
        video_channels: int = 512,
        out_channels: int = 64,
        audio_kernel: int = 4,
        video_kernel: int = 3,
        num_upsamples: int = 4, # aka q
        num_attn_heads: int = 4
    ):
        super(SeparationNetwork, self).__init__()
        self.R = R
        self.audio_rtfs = AudioProcessor(
            R=R,
            in_channels=audio_channels,
            out_channels=out_channels,
            hidden_size=out_channels,
            kernel_size=audio_kernel,
        )
        self.video_processor = VideoProcessor(
            in_channels=video_channels,
            hidden_size=out_channels,
            kernel_size=video_kernel,
            num_upsamples=num_upsamples,
            num_downsamples=num_upsamples,
        )
        self.fusion = CAF(
            in_channels_attn=audio_channels,
            in_channels_gt=video_channels,
            num_attn_heads=num_attn_heads,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        res_audio = audio
        audio = self.audio_rtfs(audio)
        video = self.video_processor(video)
        audio = self.fusion(audio, video)
        # TODO: check whether to add 1 more round
        for _ in range(self.R - 1):
            audio = self.audio_rtfs(audio + res_audio)

        return audio


class SpectralSS(nn.Module):
    """Block 3.3 from paper
    https://arxiv.org/pdf/2309.17189
    """
    
    def __init__(
        self,
        in_channels: int = 256,
    ):
        super(SpectralSS, self).__init__()
        self.in_channels=in_channels

        # TODO: maybe generate two masks right ahead?
        self.mask = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, a_r: torch.Tensor, a_0: torch.Tensor):
        m = self.mask(a_r) #

        B, C, T, F = m.shape

        m = m.view(B, 2, C//2, T, F)
        a_0 = a_0.view(B, 2, C//2, T, F)

        m_r = m[:, 0].unsqueeze(1) # real part
        m_i = m[:, 1].unsqueeze(1) # imaginary part
        E_r = a_0[:, 0].unsqueeze(1)
        E_i = a_0[:, 1].unsqueeze(1)

        z_r = m_r * E_r - m_i * E_i
        z_i = m_r * E_i + m_i * E_r

        z = torch.cat([z_r, z_i], dim=2).squeeze(1)
        return z # [B, C, T, F]


class AudioDecoder(nn.Module):
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        in_channels: int = 256,
        kernel_size: int = 3,
    ):
        super(AudioDecoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.decoder = nn.ConvTranspose2d(in_channels=in_channels, out_channels=2, kernel_size=kernel_size, padding=1, bias=False)
        self.register_buffer(
            name="hann_window",
            tensor=torch.hann_window(n_fft)
        )

    def forward(self, x: torch.Tensor):
        B, C, T, F = x.shape
        x = self.decoder(x)
        real = x[:, 0]
        complex = x[:, 1]
        complex_domain = torch.complex(real, complex).transpose(1, 2).contiguous()
        # [B, F, T]

        audio = torch.istft(
            complex_domain,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,
            length=32000,
        ) # [B, T_original]

        return audio


class RTFSNetwork(nn.Module):
    def __init__(
        self,
        R: int = 4,
        n_fft: int = 256,
        hop_length: int = 128,
        audio_out_channels: int = 256,
        video_embed_size: int = 512,
        reduced_channels: int = 64,
        audio_kernel_size: int = 4,
        video_kernel_size: int = 3,
        num_upsamples: int = 4, # TODO: maybe reduce for video?
        num_attn_heads: int = 4,
    ):
        super(RTFSNetwork, self).__init__()
        self.audio_encoder = AudioEncoder(in_channels=2, out_channels=audio_out_channels)
        self.separation_network = SeparationNetwork(
            R=R,
            audio_channels=audio_out_channels,
            video_channels=video_embed_size,
            out_channels=reduced_channels,
            audio_kernel=audio_kernel_size,
            video_kernel=video_kernel_size,
            num_upsamples=num_upsamples,
            num_attn_heads=num_attn_heads,
        )
        self.spectral_ss = SpectralSS(in_channels=audio_out_channels)
        self.audio_decoder = AudioDecoder(
            n_fft=n_fft,
            hop_length=hop_length,
            in_channels=audio_out_channels
        )

    def forward(self, complex_spectrogram: torch.Tensor, s1_embedding: torch.Tensor, s2_embedding: torch.Tensor, **batch):
        encoded_spec = self.audio_encoder(complex_spectrogram)
        # [1, 256, 251, 129]
        # speaker 1
        a_R_1 = self.separation_network(encoded_spec, s1_embedding)
        audio_complex_1 = self.spectral_ss(a_R_1, encoded_spec)
        s1_pred = self.audio_decoder(audio_complex_1)

        # speaker 2
        a_R_2 = self.separation_network(encoded_spec, s2_embedding)
        audio_complex_2 = self.spectral_ss(a_R_2, encoded_spec)
        s2_pred = self.audio_decoder(audio_complex_2)

        return {"s1_pred": s1_pred, "s2_pred": s2_pred}
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
