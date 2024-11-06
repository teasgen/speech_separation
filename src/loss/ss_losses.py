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
        super().__init__(nn.MSELoss(reduction='mean'))

    def forward(
        self,
        s1_spec_pred: torch.Tensor,
        s2_spec_pred: torch.Tensor,
        s1_spec_true: torch.Tensor,
        s2_spec_true: torch.Tensor,
        **batch
    ):
        loss_s1 = self.loss(s1_spec_pred, s1_spec_true)
        loss_s2 = self.loss(s2_spec_pred, s2_spec_true)
        total_loss = loss_s1 + loss_s2
        return {"loss": total_loss}


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
    def forward(self, s1_pred: torch.Tensor, s2_pred: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, **batch):
        return super().forward(s1_pred, s2_pred, s1, s2)


class SiSNRLoss(nn.Module):
    def __init__(self):
        super(SiSNRLoss, self).__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, **batch):
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        gt = gt - torch.mean(gt, dim=-1, keepdim=True)

        dot_product = (gt * pred).sum(dim=-1, keepdim=True)
        gt_energy = torch.linalg.norm(gt, ord=2, dim=-1, keepdim=True) ** 2
        scaling_factor = dot_product / gt_energy

        scaled_gt = scaling_factor * gt

        e_noise = pred - scaled_gt
        signal_power = torch.linalg.norm(scaled_gt, ord=2, dim=-1, keepdim=True) ** 2
        noise_power = torch.linalg.norm(e_noise, ord=2, dim=-1, keepdim=True) ** 2

        return (-20 * torch.log10(signal_power / noise_power)).mean()


class SiSNRWavLoss(BaseSSLoss):
    def __init__(self):
        loss = SiSNRLoss()
        super().__init__(loss)

    def forward(self, s1_pred: torch.Tensor, s2_pred: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, **batch):
        return super().forward(s1_pred, s2_pred, s1, s2)
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
