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
        stride: int = 2, # TODO: remove right to model
        act_type: str = "PReLU", # TODO: remove
    ):
        super(AudioNet, self).__init__()

