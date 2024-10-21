from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(fs, extended)

    def __call__(self, preds, target, **kwargs):
        return self.metric(preds, target).item()