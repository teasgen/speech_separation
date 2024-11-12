import torch
import torch.nn as nn

from .rtfs import RTFSBlock
    

class GroupConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 4,
        audio: bool = True,
        sigmoid: bool = False
    ):
        super(GroupConv, self).__init__()
        if audio:
            cnn = nn.Conv2d
        else:
            cnn = nn.Conv1d
        assert in_channels == out_channels

        self.cnn = cnn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,)
        
        self.norm = nn.InstanceNorm2d(num_features=in_channels)
        
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# see reconstruction phase from https://arxiv.org/pdf/2309.17189
class TFARUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size
    ):
        super(TFARUnit, self).__init__()
        self.w1 = GroupConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.w2 = GroupConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.w3 = GroupConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, sigmoid=True)

    def forward(self, n: torch.Tensor, m: torch.Tensor):
        term1 = self.w1(n)
        target_shape = n.shape[-(len(n.shape) // 2) :]
        term3 = nn.functional.interpolate(self.w2(m), size=target_shape, mode="nearest")
        term2 = nn.functional.interpolate(self.w3(m), size=target_shape, mode="nearest")

        upsampled = term1 * term2 + term3
        return upsampled


class RTSFNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        hidden_size: int = 64,
        kernel_size: int = 4,
    ):
        """
        Full cycle of RTFSNet block, ready to be implemented
        in the model. Downsamples by reducing the channels dimension
        and pooling, then applies the DPRNN-based block,
        then fuses the features using attention
        
        Keyword arguments:
        in_channels -- full 
        Return: return_description
        """
        
        super(RTSFNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        self.skip_connection = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.InstanceNorm2d(num_features=self.in_features),
            nn.ReLU(),)

        self.cnn = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=1
            ),
            nn.InstanceNorm2d(num_features=self.hidden_size),
            nn.ReLU(),)

        self.downsample_1 = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=4,
                stride=1,
                bias=False
            ),
            nn.InstanceNorm2d(num_features=self.hidden_size),
            nn.ReLU(),
        )

        self.downsample_2 = nn.Sequential(
            nn.InstanceNorm2d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=4,
                stride=2,
                bias=False
            ),
            nn.InstanceNorm2d(num_features=self.hidden_size),
            nn.ReLU(),
        )

        self.pooling1 = nn.AdaptiveAvgPool2d(...)
        self.pooling2 = nn.AdaptiveAvgPool2d(...)
        self.fusion_1 = ...
        self.fusion_2 = ...
        self.upsample = ...

        self.dual_path = RTFSBlock(in_channels=self.hidden_size)



# TODO: implement
class CAFBlock(nn.Module):
    def __init__(
        self,
        R: int = 4
    ):
        super(CAFBlock, self).__init__()
        self.audio_net = RTSFNet(
            in_channels=256,
            out_channels=64,
            hidden_size=64,
            kernel_size=4,
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio = self.audio_net(audio)
        video = self.video_net(video)
        audio, video = self.fusion_block(audio, video)

        return audio