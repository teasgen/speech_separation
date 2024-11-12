import torch
import torch.nn as nn


# TODO: implement
class CAFBlock(nn.Module):
    def __init__(
        self,

    ):
        super(CAFBlock, self).__init__()
        pass

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        # 1 time Fusion Block
        audio = self.audio_net(audio)
        video = self.video_net(video)

        audio, video = self.fusion_block(audio, video)

        return audio
    

class AudioNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        hidden_size: int = 64,
        kernel_size: int = 4,
        upsamples: int = 2
    ):
        super(AudioNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.upsamples = upsamples
        
        self.pooling = nn.functional.adaptive_avg_pool2d
        
        self.gate = nn.Sequential(
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.in_channels
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.in_channels
            ),
            nn.ReLU(),
        )

        self.compressor = nn.Sequential(
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.in_channels
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=1
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=self.hidden_size
            ),
            nn.ReLU(),
        )

        self.downsamples = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.concats = nn.ModuleList()

        for i in range(self.upsamples):
            self.downsamples.append(
                nn.Sequential(
                    nn.GroupNorm(
                        num_groups=1,
                        num_channels=self.hidden_size
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.hidden_size,
                        out_channels=self.hidden_size,
                        kernel_size=4,
                        stride=1 if i == 0 else 2,
                        bias=False
                    ),
                    nn.GroupNorm(
                        num_groups=1,
                        num_channels=self.hidden_size
                    ),
                    nn.ReLU(),
                )
            )

            self.fusions.append(
                # TODO: add injection fusion
            )

            self.concats.append(
                # TODO: add InjectionMultiSum
            )


        self.global_extractor = nn.Sequential(
            #TODO: add DPRNN, DPRNN, MultiHeadSelfAttention
        )


