import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk for source separation.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics.
            part (str): name of the partition, used to define the saving
                directory.
        Returns:
            batch (dict): updated batch containing model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        batch_size = batch["s1_pred"].shape[0]
        current_id = batch_idx * batch_size
        # TODO: refactor so that code below depends on model (?)
        if "s1_pred" in batch:
            # for i in range(1, 3):
                # # inverse to what was done in get_magnitude
                # # TODO: make a separate function/
                # spec = (torch.clamp(batch[f"s{i}_spec_pred"], 0.0, 1.0) - 1.0) * 100.0 + 20.0
                # spec = 10.0 ** (spec * 0.05)
                # complex_spectrum = torch.polar(spec, batch["mix_phase"])

                # batch[f"s{i}_pred"] = torch.istft(
                #     complex_spectrum,
                #     n_fft=self.n_fft,
                #     hop_length=self.hop_length,
                #     win_length=self.n_fft,
                #     center=True,
                #     window=self.window,
                # )

            if metrics is not None:
                for met in self.metrics["inference"]:
                    metrics.update(met.name, met(**batch))

                for i in range(batch_size):
                    s1_pred = batch["s1_pred"][i].clone()
                    s2_pred = batch["s2_pred"][i].clone()
                    s1_true = batch["s1"][i].clone()
                    s2_true = batch["s2"][i].clone()

                    output_id = current_id + i

                    output = {
                        "s1_pred": s1_pred,
                        "s2_pred": s2_pred,
                        "s1_true": s1_true,
                        "s2_true": s2_true,
                    }

                    if self.save_path is not None:
                        torch.save(
                            output,
                            self.save_path / part / f"output_{output_id}.pth",
                        )

                return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
