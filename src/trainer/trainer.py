import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

import pandas as pd

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        # TODO: refactor so that code below depends on model (?)
        if "s1_spec_pred" in batch:
            for i in range(1, 3):
                # inverse to what was done in get_magnitude
                # TODO: make a separate function/class for this
                spec = (torch.clamp(batch[f"s{i}_spec_pred"], 0.0, 1.0) - 1.0) * 100.0 + 20.0
                spec = 10.0 ** (spec * 0.05)
                complex_spectrum = torch.polar(
                    spec,
                    batch["mix_phase"]
                )

                batch[f"s{i}_pred"] = torch.istft(
                    complex_spectrum,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    center=True,
                    window=self.window
                )

            # MAYBE COMOUTE LOSS FOR specs?

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        self.log_audio(batch, batch_idx, mode) 

    def log_audio(self, batch, batch_idx, mode):
        audio_samples = {
            "mix": self.writer.wandb.Audio(batch["mix"][0].detach().cpu().numpy(), sample_rate=16000),
            "s1": self.writer.wandb.Audio(batch["s1"][0].detach().cpu().numpy(), sample_rate=16000),
            "s2": self.writer.wandb.Audio(batch["s2"][0].detach().cpu().numpy(), sample_rate=16000),
            "s1_pred": self.writer.wandb.Audio(batch["s1_pred"][0].detach().cpu().numpy(), sample_rate=16000),
            "s2_pred": self.writer.wandb.Audio(batch["s2_pred"][0].detach().cpu().numpy(), sample_rate=16000),
        }

        audio_df = pd.DataFrame([audio_samples]) 
        self.writer.add_table(f"{mode}/audio_table", audio_df)

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # TODO: add logging
            # Log Stuff
            pass
        else:
            # Log Stuff
            # TODO: add logging
            pass
