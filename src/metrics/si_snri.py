from src.metrics.base_metric import SS2BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import torch

class SISNRiMetric(SS2BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalNoiseRatio().to(kwargs.get("device", "cpu"))

    def __call__(self, s1_pred: torch.Tensor, s2_pred: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, mix: torch.Tensor, **batch):
        si_snr_sep = self.forward(s1_pred=s1_pred, s2_pred=s2_pred, s1=s1, s2=s2, **batch)

        si_snr_mix_s1 = self.metric(s1, mix)
        si_snr_mix_s2 = self.metric(s2, mix)

        si_snr_mix = (si_snr_mix_s1 + si_snr_mix_s2) / 2

        return si_snr_sep - si_snr_mix