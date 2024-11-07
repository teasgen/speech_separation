from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import SS2BaseMetric


class SISNRMetric(SS2BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalNoiseRatio().to(
            kwargs.get("device", "cpu")
        )

    def __call__(self, **batch):
        return super().forward(**batch)
