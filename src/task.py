import random
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Any

import torch
from beartype import beartype
from lightning import LightningModule
from torch import Tensor

from model import CausalEncoder, CausalDecoder
class my_custom_task(ABC, LightningModule):
    @beartype
    def __init__(
        self,
        encoder : CausalEncoder,
        decoder : CausalDecoder,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

    @beartype
    def forward(self, x: Any) -> Any:
        """Return the training predictions, next-token predictions, and aggregated context z.

        Args:
            x (dict[str, Tensor]):  Input data, each with shape (samples, tasks, *).

        Returns:
            any: Predictions.
        """
        pass

    @beartype
    def training_step(self, data, batch_idx):
        pass

    @beartype
    def validation_step(self, data, batch_idx):
        pass

    @torch.inference_mode()
    def on_train_end(self):
        pass

    @abstractmethod
    def loss_function(self, target: Any, preds: Any) -> Tensor:
        """Do not average across samples and tasks! Return shape should be

        Args:
            target (dict[str, Tensor]): Inputs/targets (samples, tasks, *).
            preds (dict[str, Tensor]): Predictions (samples, tasks, *).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    """example of property that can be added
    @property
    def is_a_lightning_module(self) -> bool:
        return True
    """
