import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Literal, Tuple

import torch
from beartype import beartype
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F

from model import CMDecoder, CMEncoder


class SCM_Reconstruction(ABC, LightningModule):
    """VAE for causal modeling."""

    @beartype
    def __init__(
        self,
        encoder: CMEncoder,
        decoder: CMDecoder,
        temperature: float,
        lr: float = 1e-4,
    ):
        self.save_hyperparameters(ignore="encoder, decoder")
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.beta = nn.Parameter(torch.tensor(1.0))  # Learnable beta parameter

    @beartype
    def gumbel_softmax_sample(self, logits: Tensor, temperature: float) -> Tensor:
        gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        gumbels = (logits + gumbels) / temperature  # Add Gumbel noise and divide by temperature
        return F.softmax(gumbels, dim=2)

    @beartype
    def forward(self, x: Tensor) -> Dict:
        """Return the predicted mu_hat of the gaussian model of the likelihood.

        Args:
            data : Input data containing x and target with the shape (batch, *). #TODO verify dimensions

        Returns:
            any: Predictions.
        """

        unormalized_dist = self.encoder(x)
        unormalized_dist = torch.stack([unormalized_dist, 1 - unormalized_dist], dim=2)
        posterior = self.gumbel_softmax_sample(unormalized_dist, temperature=self.temperature)
        z = torch.argmax(posterior, dim=2)
        # straight through gradient estimator
        z = (z - posterior[:, :, 1]).detach() + posterior[:, :, 1]
        mu_hat = self.decoder(z)
        return {"mu_hat": mu_hat, "posterior": posterior, "z": z}

    @beartype
    def training_step(self, data: Any, batch_idx: int) -> Tensor:
        x, target = data
        dict = self.forward(x)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        loss = self.loss_function(x, mu_hat, posterior)
        self.log("train_loss", loss)
        return loss

    @beartype
    def validation_step(self, data: Any, batch_idx: int) -> Tensor:
        x, target = data
        dict = self.forward(x)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        z = dict["z"]
        loss = self.loss_function(x, mu_hat, posterior)
        self.log("val_loss", loss)
        # log beta parameter
        print(self.beta.item())
        return loss

    def loss_function(self, x, mu_hat, posterior) -> Tensor:
        """

        Returns:
            Tensor: Losses (samples, tasks).
        """
        # elementwise squared error
        # MSE loss between the input and the output
        reco_loss = (x - mu_hat).pow(2).mean()
        self.log("reconstruction_loss", reco_loss)

        regularization_loss = 0

        entropy = -(
            posterior[:, :, 0] * torch.log(posterior[:, :, 0])
            + posterior[:, :, 1] * torch.log(posterior[:, :, 1])
        )
        entropy = entropy[torch.isfinite(entropy)].view_as(entropy)

        regularization_loss += torch.mean(torch.sum(entropy, dim=1))

        self.log("regularization_loss", regularization_loss)
        loss = reco_loss + self.beta * regularization_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    """example of property that can be added
    @property
    def is_a_lightning_module(self) -> bool:
        return True
    """
