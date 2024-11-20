import torch
import torch.nn as nn
import torch.nn.functional as F


# https://arxiv.org/pdf/2002.08688
class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, kernel_size=32, stride=16, bias=False):
        super(Encoder, self).__init__()
        self.L = stride
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=16),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.pad(x, (self.L, self.L*2), mode='constant', value=0)
        return self.sequential(x)
    

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
        self.N = N
        self.norm_1 = GlobalNorm(N)
        self.conv1d = nn.Conv1d(N, B, 1)
        self.separator = nn.ModuleList([Conv1D_Block(B, H, R, dilation=2**i)
                                        for s in range(P) for i in range(X)])
        self.seq = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N * 2, 1))

    def forward(self, x):
        masked_x = x.unsqueeze(1)
        x = self.norm_1(x)
        x = self.conv1d(x)
        acc_sc = 0.0
        
        for layer in self.separator:
            residual, score = layer(x)
            x = x + residual
            acc_sc = acc_sc + score
        
        acc_sc = self.seq(acc_sc)
        acc_sc = torch.sigmoid(acc_sc)
        acc_sc = acc_sc.reshape(acc_sc.shape[0], 2, self.N, -1)

        x = masked_x * acc_sc
        x = x.reshape(-1, self.N, x.shape[-1])

        return x

class Decoder(nn.Module):
    def __init__(self, N=512, out_channels=1, kernel_size=32, stride=16, bias=False):
        super(Decoder, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(N, out_channels, kernel_size=kernel_size, stride=stride, bias=True)
        )
        self.deconv = nn.ConvTranspose1d(N, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x, batch_size):
        x = self.sequential(x)
        x = torch.index_select(
            x, 2, torch.arange(self.stride, x.shape[2] - self.kernel_size, device=x.device)
        )
        x = x.reshape(batch_size, 2, -1)
        return x

class DeepAVConvTasNet(nn.Module):
    def __init__(
            self, 
            N=512, 
            L=16,
            video_emb_size=512,
            hidden_video=512):
        super().__init__()
        self.encoder = Encoder()
        self.separator = Separator()
        self.decoder = Decoder()
        self.visual_compression = nn.Linear(video_emb_size, hidden_video // 2)
        self.video_ln = nn.LayerNorm(hidden_video)

    def forward(self, mix, s1_embedding, s2_embedding, **batch):
        batch_size = mix.shape[0]
        mix = self.encoder(mix)

        s1_embedding = self.visual_compression(
            s1_embedding.permute(0, 2, 1)
        )
        s2_embedding = self.visual_compression(
            s2_embedding.permute(0, 2, 1)
        ) 

        video = torch.concat([s1_embedding, s2_embedding], -1)
        video = F.interpolate(
            video.permute(0, 2, 1), size=mix.shape[-1], mode="linear", align_corners=False
        ).permute(0, 2, 1)
        video = self.video_ln(video).permute(0, 2, 1)

        mix = mix + video
        mix = self.separator(mix)
        mix = self.decoder(mix, batch_size)
        return {"s1_pred": mix[:, 0], "s2_pred": mix[:, 1]}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info