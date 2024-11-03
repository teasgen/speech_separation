from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch
import pandas as pd

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_audio_samples = []
        self.step = 0

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        self.step += 1
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
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