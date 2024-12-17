from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F

from dataset import SCM


class abstract_model(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass


class CMEncoder(abstract_model):
    @beartype
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        activation: str = "relu",
    ) -> None:
        super(CMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()

        self.model = build_model(
            self.input_dim, self.hidden_dim, self.latent_dim, self.n_layers, self.activation
        )
        # Move the model to the specified device if provided

    @beartype
    def forward(self, data: Tensor) -> Tensor:
        """forward pass of the encoder."""
        return self.model(data)

    @beartype
    def to(self, device):
        super().to(device)
        # TODO check if this is necessary
        return self


class CMDecoder(abstract_model):
    """Decoder for the Causal Model. The idea is to simulate the data reconstruction from noise autoregressively given the latent representation that's meant to represent the interventional regime"""

    @beartype
    def __init__(
        self,
        graph: SCM,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "relu",
    ) -> None:
        super(CMDecoder, self).__init__()
        self.graph = graph
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        self.topological_orders = self.graph.get_topological_orders()
        self.mechanisms = nn.ModuleList()
        for i in range(self.graph().shape[0]):
            numb_parents = self.graph()[:, self.topological_orders[i]].sum()
            input_dim = 2 + numb_parents
            self.mechanisms.append(build_model(input_dim, self.hidden_dim, 1, self.n_layers, self.activation))

    @beartype
    def forward(self, z: Tensor) -> Tensor:
        """Forward pass of the decoder."""
        x = torch.zeros(z.shape[0], self.graph().shape[0]).type_as(z)
        # permute z to match the topological order
        for i, mechanism in enumerate(self.mechanisms):
            variable = self.topological_orders[i]
            parents = self.graph()[:, variable].nonzero()[0]

            input = torch.cat(
                (
                    x[:, parents],
                    z[:, self.topological_orders[i] + 1].unsqueeze(1),
                    1 - z[:, self.topological_orders[i] + 1].unsqueeze(1),
                ),
                dim=1,
            )  # todo verify if that suffices
            x[:, self.topological_orders[i]] = mechanism(input).flatten()

        return x


def build_model(
    input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, activation: nn.Module
) -> nn.Sequential:
    layers = [
        nn.Linear(input_dim, hidden_dim),
        activation,
    ]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation)
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)
