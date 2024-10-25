from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode='wb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, preds, target, **kwargs):
        return self.metric(preds, target).item()