import torch
import torch.nn as nn

class AudioProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        kernel_size: int = 1
    ):
        super(AudioProcessor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.net = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                bias=True
            ),
            nn.InstanceNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)