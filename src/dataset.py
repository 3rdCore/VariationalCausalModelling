import copy
from abc import ABC,abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

from lightning import LightningDataModule

warnings.filterwarnings("ignore", message=".*does not have many workers.*") 

class Dataset(ABC, MapDataPipe):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get sample from the dataset.

        Args:
            index (int): sample index.

        Returns:
            Any: sample.
        """
        pass

    @beartype
    def __len__(self) -> int:
        pass

    @abstractmethod
    def gen_data(
        self,
        n_samples: int,
    ) -> Any:
        pass


class Custom_Dataset(Dataset):
    @beartype
    def __init__(
        self,
    ):
        super().__init__()

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
        n_samples: int,
    ) -> Any:
        x = None
        y = None
        data_dict = {"x": x, "y": y}
        params_dict = None
        
        return data_dict, params_dict


    @abstractmethod
    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        """Sample parameters for each of the n_tasks

        Args:
            n_tasks (int): Number of tasks to generate.

        Returns:
            dict[str, Tensor]:
        """
        pass

    @abstractmethod
    def function(self, x: Tensor, params: dict[str, Tensor]) -> FloatTensor:
        """Applies the function defined using the params (function parameters)
        on the x(input data) to get y (output)

        Args:
            x (Tensor): input data with shape (n_tasks, n_samples, x_dim)
            params (int): function parameters with shape (n_tasks, ...)

        Returns:
            FloatTensor: y (output) with shape (n_tasks, n_samples, y_dim)
        """
        pass


class CustomDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # setup

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

