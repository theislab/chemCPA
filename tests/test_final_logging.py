import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch
from lightning.pytorch.loggers.logger import Logger
from torch.utils.data import DataLoader, TensorDataset

from chemCPA.lightning_module import ChemCPA


class RecordingLogger(Logger):
    def __init__(self):
        super().__init__()
        self.metrics = []

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return "0"

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))


class FinalEvaluationModel(ChemCPA):
    def __init__(self, run_eval=True):
        L.LightningModule.__init__(self)
        self.model = torch.nn.Linear(1, 1)
        self.config = {"training": {"run_eval_r2": run_eval}}

    def training_step(self, batch, batch_idx):
        return self.model(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_validation_epoch_end(self):
        pass


class EvaluationData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        for split in ("train", "test", "ood"):
            setattr(self, f"{split}_treated_dataset", object())
            setattr(self, f"{split}_control_dataset", SimpleNamespace(genes=None))

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)


class TestFinalLogging(unittest.TestCase):
    def fit_model(self, logger, run_eval=True):
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        with patch(
            "chemCPA.lightning_module.evaluate_r2",
            side_effect=[[0.1], [0.2], [0.3]],
        ) as evaluate:
            trainer.fit(FinalEvaluationModel(run_eval), datamodule=EvaluationData())
        return trainer, evaluate

    def test_final_metrics_reach_every_logger(self):
        loggers = [RecordingLogger(), RecordingLogger()]
        trainer, evaluate = self.fit_model(loggers)
        expected = {
            "final_train_R2_mean": 0.1,
            "final_test_R2_mean": 0.2,
            "final_ood_R2_mean": 0.3,
        }
        self.assertEqual(evaluate.call_count, 3)
        for logger in loggers:
            self.assertIn((expected, trainer.global_step), logger.metrics)

    def test_training_finishes_without_a_logger(self):
        _, evaluate = self.fit_model(False)
        self.assertEqual(evaluate.call_count, 3)

    def test_disabled_evaluation_does_not_log_final_metrics(self):
        logger = RecordingLogger()
        _, evaluate = self.fit_model(logger, run_eval=False)
        evaluate.assert_not_called()
        self.assertFalse(any("final_train_R2_mean" in m for m, _ in logger.metrics))
