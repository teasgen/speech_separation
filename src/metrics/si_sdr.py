from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from src.metrics.base_metric import SS2BaseMetric


class SISDRMetric(SS2BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio().to(
            kwargs.get("device", "cpu")
        )

    def __call__(self, **batch):
        return super().forward(**batch)
