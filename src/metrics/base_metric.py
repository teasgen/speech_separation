from abc import abstractmethod

import torch


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


class SS2BaseMetric:
    """
    Base class for two SS metrics
    """

    def __init__(self, name=None, lower_better=False, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__
        self.result_function = min if lower_better else max

    @abstractmethod
    def forward(
        self,
        s1_pred: torch.Tensor,
        s2_pred: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        **batch
    ):
        """
        Defines PIT metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        s1_s1 = self.metric(s1_pred, s1).item()
        s2_s1 = self.metric(s2_pred, s1).item()
        s1_s2 = self.metric(s1_pred, s2).item()
        s2_s2 = self.metric(s2_pred, s2).item()

        perm_1_score = (s1_s1 + s2_s2) / 2
        perm_2_score = (s1_s2 + s2_s1) / 2
        return self.result_function(perm_1_score, perm_2_score)
