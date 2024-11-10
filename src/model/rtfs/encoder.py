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
            bias=False
        )

        gln1 = nn.GroupNorm(
            num_groups=1,
            num_channels=self.in_channels
        )

        gln2 = nn.GroupNorm(
            num_groups=1,
            num_channels=self.out_channels
        )

        self.net = nn.Sequential(
            gln1,
            nn.ReLU(),
            cnn,
            gln2,
            nn.ReLU()
        )

    def forward(self, mix: torch.Tensor):
        alpha = torch.stft(
            mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True
        )
        # NOTE: spec is transposed! to follow paper notation
        complex_domain = torch.stack(
            [alpha.real, alpha.imag],
            dim = 1
        ).transpose(2, 3).contiguous() # [B, 2, T, F]
        
        embedding = self.net(complex_domain) # [B, C_out, T, F]

        return embedding
