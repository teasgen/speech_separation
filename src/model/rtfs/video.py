import torch
import torch.nn as nn


# TODO: maybe make a complex net
# lipreader is good though
class VideoProcessor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512
    ):
        super(VideoProcessor, self).__init__()
        self._embedding_dim = embedding_dim
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    @property
    def embedding_dim(self):
        return self._embedding_dim
