import sys
#sys.path.append('../')

import torch.nn.functional as F
from torch import nn
import torch
#from utils.util import check_parameters

import warnings

#warnings.filterwarnings('ignore')

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class AudioEncoder(nn.Module):
    '''
       Conv-Tasnet AudioEncoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, interm_out_channels=64, out_channels=256, norm='ln'):
        super(AudioEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=interm_out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)
        
        self.audio_norm = select_norm(norm, interm_out_channels, 3)
        self.sep_audio_conv1d = nn.Conv1d(interm_out_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        ae = F.relu(x)

        # [B, N, L]
        x = self.audio_norm(ae)
        # [B, N, L]
        x = self.sep_audio_conv1d(x)
        return ae, x


class VideoEncoder(nn.Module):
    '''
       Conv-Tasnet AudioEncoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, interm_out_channels=64, out_channels=256, norm='ln'):
        super(VideoEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=interm_out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)
        
        self.prelu = nn.PReLU()
        self.video_norm = select_norm(norm, interm_out_channels, 3)
        self.sep_video_conv1d = nn.Conv1d(interm_out_channels, out_channels, 1, bias=False)
    
    def forward(self, x, upsample_size=15999):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x FPS x VideoFeatures -> B x 1 x FPS * VideoFeatures
        x = x.flatten(start_dim=1).unsqueeze(1)
        # B x 1 x FPS * VideoFeatures -> B x C x T_out
        x = self.conv1d(x)
        x = self.prelu(x)
        # B x C x T_out -> B x C x upsample_size
        x = F.interpolate(x, upsample_size, mode='linear', align_corners=False)

        x = self.video_norm(x)
        # [B, N, L]
        x = self.sep_video_conv1d(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):#, input_lengths):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        # intra_rnn= nn.utils.rnn.pack_padded_sequence(
        #     intra_rnn, input_lengths, batch_first=True, enforce_sorted=False)
        self.intra_rnn.flatten_parameters()
        intra_rnn, _ = self.intra_rnn(intra_rnn)

        # intra_rnn = nn.utils.rnn.pad_packed_sequence(
        #     intra_rnn, batch_first=True)

        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]

        # inter_rnn= nn.utils.rnn.pack_padded_sequence(
        #     inter_rnn, input_lengths, batch_first=True, enforce_sorted=False)
        self.inter_rnn.flatten_parameters()
        inter_rnn, _ = self.inter_rnn(inter_rnn)

        # inter_rnn = nn.utils.rnn.pad_packed_sequence(
        #     inter_rnn, batch_first=True)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class GlobalNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalNorm, self).__init__()
        self.register_parameter('gamma', nn.Parameter(torch.ones(channel_size, 1)))
        self.register_parameter('beta', nn.Parameter(torch.zeros(channel_size, 1)))
        self.eps = 5e-6

    def forward(self, f):
        expected_f = f.mean(dim=(1, 2), keepdim=True)
        var_f = ((f - expected_f) ** 2).mean(dim=(1, 2), keepdim=True)
        f = self.gamma * (f - expected_f) / torch.sqrt(var_f + self.eps) + self.beta
        return f
    
class Conv1D_Block(nn.Module):
    def __init__(self, input_size, hidden_size, kernel, dilation=1):
        super(Conv1D_Block, self).__init__()
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = nn.GroupNorm(1, hidden_size, eps=1e-10)
        self.dconv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel, 
                                 groups=hidden_size, padding=(dilation * (kernel - 1)) // 2, 
                                 dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = nn.GroupNorm(1, hidden_size, eps=1e-10)
        self.conv = nn.Conv1d(hidden_size, input_size, kernel_size=1)
        self.conv_sc = nn.Conv1d(hidden_size, input_size, kernel_size=1)

    def forward(self, x):
        c = self.conv1d(x)
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        c = self.dconv1d(c)
        c = self.PReLU_2(c)
        c = self.norm_2(c)
        return self.conv(c), self.conv_sc(c)

class Separator(nn.Module):
    def __init__(self, N=512, B=128, H=512, X=8, P=3, R=3):
        super().__init__()
        self.norm_1 = GlobalNorm(N)
        self.conv1d = nn.Conv1d(N, B, 1)
        self.separator = nn.ModuleList([Conv1D_Block(B, H, R, dilation=2**i)
                                        for s in range(P) for i in range(X)])
        self.seq = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N * 2, 1))

    def forward(self, x):
        x = self.norm_1(x)
        x = self.conv1d(x)
        acc_sc = 0.0
        
        for layer in self.separator:
            residual, score = layer(x)
            x += residual
            acc_sc += score
        
        return self.seq(acc_sc)

class Dual_RNN_model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2, video_kernel_size=8, include_video_features=True):
        super(Dual_RNN_model,self).__init__()
        self.include_video_features = include_video_features
        self.audio_encoder = AudioEncoder(kernel_size=kernel_size, interm_out_channels=in_channels, out_channels=out_channels, norm=norm)
        if self.include_video_features:
            self.video_encoder = VideoEncoder(kernel_size=video_kernel_size, interm_out_channels=in_channels, out_channels=out_channels, norm=norm)
            self.audio_video_projection = nn.Linear(in_features=2, out_features=1)
        
        self.separation = Separator()
        self.decoder = Decoder(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.num_spks = num_spks
    
    def forward(self, mix, s1_embedding, s2_embedding, **batch):#, input_lenghts):
        '''
           x: [B, L]
        '''
        
        # [B, N, L]
        s1_embedding = s1_embedding.permute(0, 2, 1)
        s2_embedding = s2_embedding.permute(0, 2, 1)

        ae, x = self.audio_encoder(mix)
        if self.include_video_features:
            ve = self.video_encoder(torch.cat((s1_embedding, s2_embedding), dim=-1), ae.size(-1))

        x = self.audio_video_projection(torch.stack((x, ve), dim=-1)).squeeze(-1) if self.include_video_features else x
        # [spks, B, N, L]
        s = self.separation(x)#, input_lenghts)
        # [B, N, L] -> [B, L]
        out = [s[i]*ae for i in range(self.num_spks)]
        audio = [self.decoder(out[i]) for i in range(self.num_spks)]
        return {"s1_pred": audio[0], "s2_pred": audio[1]}
