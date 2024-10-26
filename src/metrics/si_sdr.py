from src.metrics.base_metric import SS2BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDRMetric(SS2BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio().to(kwargs.get("device", "cpu"))

    def __call__(self, **batch):
        return super().forward(**batch)