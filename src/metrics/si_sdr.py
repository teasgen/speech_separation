from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio()

    def __call__(self, preds, target, **kwargs):
        return self.metric(preds, target).item()