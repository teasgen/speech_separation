import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, kernel_size=32, stride=16, bias=False):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        return self.conv1d(x)
    

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
        self.conv_block = nn.ModuleList([Conv1D_Block(B, H, R, dilation=2**i)
                                        for s in range(P) for i in range(X)])
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(B, 2*N, 1))

    def forward(self, x):
        x = self.norm_1(x)
        x = self.conv1d(x)

        acc_sc = sum(layer(x)[1] for layer in self.conv_block)
        return self.output(acc_sc)

class Decoder(nn.Module):
    def __init__(self, N=512, out_channels=1, kernel_size=32, stride=16, bias=False):
        super(Decoder, self).__init__()
        self.deconv = nn.ConvTranspose1d(N, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        return self.deconv(x)

class ConvTasNet(nn.Module):
    def __init__(self, N=512, L=16):
        super().__init__()
        self.N = N
        self.L = L
        self.encoder = Encoder()
        self.separator = Separator()
        self.decoder = Decoder()

    def forward(self, mix, **batch):

        pad = (32 - (self.L + mix.shape[1] % 32) % 32) % 32
        mix = mix.unsqueeze(1)
        mix = F.pad(mix, (self.L, pad + self.L), mode='constant', value=0)
        mix_shape = mix.shape
        mix = self.encoder(mix)

        masks = self.separator(mix)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(mix_shape[0], 2, self.N, -1)
        
        s_pred = mix.unsqueeze(1)
        s_pred = s_pred * masks
        s_pred = s_pred.reshape(-1, self.N, mix.shape[-1])
        s_pred = self.decoder(s_pred)

        device = s_pred.device
        s_pred = torch.index_select(
            s_pred, 2, torch.arange(self.L, s_pred.shape[2] - (pad + self.L), device=device)
        )
        s_pred = s_pred.reshape(mix_shape[0], 2, -1)
        
        return {"s1_pred": s_pred[:, 0], "s2_pred": s_pred[:, 1]}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
