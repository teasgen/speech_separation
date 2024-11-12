import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
            self,
            n_fft: int = 256,
            hop_length: int = 128,
            in_channels: int=2,
            out_channels: int = 256
    ):
        super(AudioEncoder, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        cnn = nn.Conv2d(
            in_channels=2,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,)

        norm1 = nn.InstanceNorm2d(
            num_features=self.in_channels,
            affine=True,)

        norm2 = nn.InstanceNorm2d(
            num_features=self.out_channels,
            affine=True,)

        self.net = nn.Sequential(
            norm1,
            nn.ReLU(),
            cnn,
            norm2,
            nn.ReLU(),)

    def forward(self, mix: torch.Tensor):
        alpha = torch.stft(
            mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True,)
        # NOTE: spec is transposed! to follow paper notation
        complex_domain = torch.stack(
            [alpha.real, alpha.imag],
            dim = 1
        ).transpose(2, 3).contiguous() # [B, 2, T, F]
        
        embedding = self.net(complex_domain) # [B, C_out, T, F]

        return embedding
