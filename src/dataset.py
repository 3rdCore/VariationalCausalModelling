from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import warnings
import networkx as nx
import causaldag as cd

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


class DAG:
    def __init__(
        self,
        n_variables: int,
        density: float,
    ):
        self.adjacency_matrix = cd.rand.directed_erdos(n_variables, density).to_amat()[
            0
        ]

    # call this class as a function
    def __call__(self):
        return self.adjacency_matrix

    def get_partial_orders(self) -> list:
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)

        # Check if the graph is a DAG
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The provided graph is not a DAG")

        # Perform topological sort to get the partial orders
        partial_orders = list(nx.topological_sort(G))
        return partial_orders


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


class SCM_Dataset(Dataset):
    @beartype
    def __init__(
        self,
        n_samples: int,
        graph: DAG,  # adjacency matrix
        observational_density: float,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.graph = graph
        self.n_variables = self.graph.shape[0]

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
    ) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Tensor]]]:

        params_dict = self.sample_params()
        partial_orders = self.graph.get_partial_orders(self.graph)

        boolean_tensor = torch.bernoulli(
            torch.full((self.n_samples,), 1 - self.observational_density)
        )
        # target are randint from 0 to n_variables + 1 : 0 is the null value
        target = boolean_tensor * torch.randint(
            1, self.n_variables + 1, (self.n_samples,)
        )

        x = torch.randn(self.n_samples, self.n_variables)
        for order in partial_orders:
            x[:, order] = self.mechanism(x, params_dict, order, target)

        data_dict = {"x": x, "target": target}
        return data_dict, params_dict

    @beartype
    def sample_params(self) -> dict[str, Tensor]:
        """Sample parameters for the SCM

        Returns:
            dict[str, Tensor]:
        """
        # sample from Unif[-1, -0.1]u[0.1, 1]
        weights = sample_weights_from_trunc_uniform(
            (self.n_variables, self.n_variables)
        )
        weights = weights * self.graph

        sigmas = torch.randn(self.n_variables)

        int_mean_shift = torch.randn(self.n_variables + 1)
        int_mean_shift[0] = 0
        int_cov_shift = torch.randn(self.n_variables + 1)
        int_cov_shift[0] = 0
        
        return {
            "weights": weights,
            "sigmas": sigmas,
            "int_mean_shift": int_mean_shift,
            "int_cov_shift": int_cov_shift,
        }

    @beartype
    def mechanism(
        self, x: Tensor, params_dict: dict[str, Tensor], order: int, target: Tensor
    ) -> FloatTensor:
        """sample value for a certain node X_i using the params defined for a certain family of mechanisms and the value oif the previouslly sampled nodes Parent(X_i)

        Returns:
            FloatTensor: y (output) with shape (n_tasks, n_samples, y_dim)
        """
        mu = torch.matmul(x, params_dict["weights"][:, order])+ params_dict["int_mean_shift"][target]
            
        sigma = params_dict["sigmas"][order] + params_dict["int_cov_shift"][target]
        #sample from normal distribution
        
        x[:, order] = mu + sigma*torch.randn(self.n_samples) 

        return x[:, order]


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


def sample_weights_from_trunc_uniform(size: tuple) -> np.ndarray:
    # Sample from Unif[-1, -0.1]
    samples1 = np.random.uniform(-1, -0.1, size)
    # Sample from Unif[0.1, 1]
    samples2 = np.random.uniform(0.1, 1, size)

    # Randomly choose between samples1 and samples2
    choices = np.random.choice([0, 1], size=size)
    samples = np.where(choices == 0, samples1, samples2)

    return samples
