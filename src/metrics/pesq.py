from src.metrics.base_metric import SS2BaseMetric
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


class PESQMetric(SS2BaseMetric):
    def __init__(self, fs=16000, mode='wb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode).to(kwargs.get("device", "cpu"))

    def __call__(self, **batch):
        return super().forward(**batch)