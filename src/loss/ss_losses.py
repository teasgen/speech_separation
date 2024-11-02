import torch
from torch import nn


class BaseSSLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(
        self,
        s1_pred: torch.Tensor,
        s2_pred: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        **batch
    ):
        """
        Applies PIT on self.loss
        """
        loss_perm_1 = (self.loss(s1_pred, s1) + self.loss(s2_pred, s2)) / 2
        loss_perm_2 = (self.loss(s1_pred, s2) + self.loss(s2_pred, s1)) / 2
        loss = loss_perm_1
        if loss_perm_2 < loss_perm_1:
            loss = loss_perm_2
        return {"loss": loss}


class MSESpecLoss(BaseSSLoss):
    def __init__(self):
        loss = nn.MSELoss()
        super().__init__(loss)

    def forward(
        self,
        s1_pred: torch.Tensor,
        s2_pred: torch.Tensor,
        s1_spectrogram: torch.Tensor,
        s2_spectrogram: torch.Tensor,
        **batch
    ):
        return super().forward(s1_pred, s2_pred, s1_spectrogram, s2_spectrogram)


class MSEWavLoss(BaseSSLoss):
    def __init__(self):
        loss = nn.MSELoss()
        super().__init__(loss)

    def forward(
        self,
        s1_pred: torch.Tensor,
        s2_pred: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        **batch
    ):
        return super().forward(s1_pred, s2_pred, s1, s2)


class MAEWavLoss(BaseSSLoss):
    def __init__(self):
        loss = nn.L1Loss()
        super().__init__(loss)

    def forward(
        self,
        s1_pred: torch.Tensor,
        s2_pred: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        **batch
    ):
        return super().forward(s1_pred, s2_pred, s1, s2)


# TODO: add custom loss for VoiceFilter (?)
