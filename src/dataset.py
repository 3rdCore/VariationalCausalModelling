import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import causaldag as cd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class DAG:
    def __init__(
        self,
        n_variables: int,
        density: float,
    ):
        self.adjacency_matrix = cd.rand.directed_erdos(n_variables, density).to_amat()[0]

    # call this class as a function
    def __call__(self):
        return self.adjacency_matrix

    def get_topological_orders(self) -> list:
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)

        # Check if the graph is a DAG
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The provided graph is not a DAG")

        # Perform topological sort to get the partial orders
        topological_orders = list(nx.topological_sort(G))
        return topological_orders


class Dataset(ABC):
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
    generator_type = Literal["LinearANM", "LinearGaussian"]

    @beartype
    def __init__(
        self,
        n_samples: int,
        graph: DAG,  # adjacency matrix
        observational_density: float,
        generator: generator_type,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.graph = graph
        self.observational_density = observational_density
        self.generator = generator
        self.n_variables = self.graph().shape[0]
        self.data, self.task_params = self.gen_data()

    @beartype
    def __len__(self) -> int:
        return self.n_samples

    @beartype
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        data_tuple = (self.data["x"][index], self.data["target"][index])
        return data_tuple

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
    ) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Tensor]]]:

        params_dict = self.sample_params()
        topological_orders = self.graph.get_topological_orders()

        boolean_tensor = torch.bernoulli(torch.full((self.n_samples,), 1 - self.observational_density))
        # target are randint from 0 to n_variables + 1 : 0 is the null value
        target = (boolean_tensor * torch.randint(1, self.n_variables + 1, (self.n_samples,))).int()

        x = torch.randn(self.n_samples, self.n_variables, dtype=torch.float32)  # normal exogenous noise
        for order in topological_orders:
            self.mechanism(x, params_dict, order, target)

        data_dict = {"x": x, "target": target}
        return data_dict, params_dict

    @beartype
    def sample_params(self) -> dict[str, Tensor]:
        """Sample parameters for the SCM

        Returns:
            dict[str, Tensor]: Dictionary containing the parameters of the SCM.
        """
        # sample from Unif[-1, -0.1]u[0.1, 1]
        weights = sample_weights_from_trunc_uniform((self.n_variables, self.n_variables))
        weights = weights * torch.from_numpy(self.graph())

        sigmas = sample_weights_from_trunc_uniform((self.n_variables,))

        int_mean_shift = sample_weights_from_trunc_uniform((self.n_variables + 1,))
        int_mean_shift[0] = 0
        int_cov_shift = sample_weights_from_trunc_uniform((self.n_variables + 1,))
        int_cov_shift[0] = 0

        return {
            "weights": weights,
            "sigmas": sigmas,
            "int_mean_shift": int_mean_shift,
            "int_cov_shift": int_cov_shift,
        }

    @beartype
    def mechanism(self, x: Tensor, params_dict: dict[str, Tensor], order: int, target: Tensor) -> None:
        """sample value for a certain node X_i using the params defined for a certain family of mechanisms and the value oif the previouslly sampled nodes Parent(X_i)

        Returns:
            FloatTensor: y (output) with shape (n_tasks, n_samples, y_dim)
        """
        weights = params_dict["weights"]
        is_node_intervened = torch.where((target - 1 == order), torch.tensor(order) + 1, torch.tensor(0))

        mu = (
            x[:, order]
            + torch.matmul(x, weights[:, order])
            + params_dict["int_mean_shift"][is_node_intervened]
        )

        sigma = params_dict["sigmas"][order] + params_dict["int_cov_shift"][is_node_intervened]

        # Sample from normal distribution
        if self.generator == "LinearANM":
            x[:, order] = mu
        elif self.generator == "LinearGaussian":
            x[:, order] = mu + sigma * torch.randn_like(mu)
        else:
            raise ValueError("Invalid generator type")


class SCMDataModule(LightningDataModule):
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


def sample_weights_from_trunc_uniform(size: tuple) -> Tensor:
    # Sample from Unif[-1, -0.1]
    samples1 = torch.FloatTensor(size=size).uniform_(-1, -0.1)
    # Sample from Unif[0.1, 1]
    samples2 = torch.FloatTensor(size=size).uniform_(0.1, 1)

    # Randomly choose between samples1 and samples2
    choices = torch.randint(0, 2, size)
    samples = torch.where(choices == 0, samples1, samples2)

    return samples.to(torch.float32)
