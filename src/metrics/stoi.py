from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import SS2BaseMetric


class STOIMetric(SS2BaseMetric):
    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(fs, extended).to(
            kwargs.get("device", "cpu")
        )

    def __call__(self, **batch):
        return super().forward(**batch)
