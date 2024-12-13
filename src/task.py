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
        temperature: float | int,
        is_beta_VAE: bool,
        lr: float = 1e-4,
        beta: float | int = 1.0,
    ):
        self.save_hyperparameters(ignore="encoder, decoder")
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=is_beta_VAE)

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
        # sigmoid activation to get the probability of the latent variable
        posterior = torch.sigmoid(self.encoder(x))
        posterior = torch.stack([posterior, 1 - posterior], dim=2)
        z_float = self.gumbel_softmax_sample(posterior, temperature=self.temperature)
        z_hard = torch.argmax(z_float, dim=2)
        # straight through gradient estimator
        z = (z_hard - z_float[:, :, 1]).detach() + z_float[:, :, 1]
        mu_hat = self.decoder(z)
        return {"mu_hat": mu_hat, "posterior": posterior, "z": z}

    @beartype
    def training_step(self, data: Any, batch_idx: int) -> Tensor:
        x, mu, target = data
        dict = self.forward(x)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        z = dict["z"]
        loss = self.loss_function(mu, mu_hat, posterior)
        self.log("train_loss", loss)
        return loss

    @beartype
    def validation_step(self, data: Any, batch_idx: int) -> Tensor:
        x, mu, target = data
        dict = self.forward(x)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        z = dict["z"]
        loss = self.loss_function(mu, mu_hat, posterior)
        self.log("val_loss", loss)
        return loss

    def loss_function(self, mu, mu_hat, posterior) -> Tensor:
        """

        Returns:
            Tensor: Losses (samples, tasks).
        """
        # elementwise squared error
        # MSE loss between the input and the output
        reco_loss = F.mse_loss(mu, mu_hat, reduction="sum")
        self.log("reconstruction_loss", reco_loss)

        regularization_loss = 0

        entropy = posterior[:, :, 0] * torch.log(posterior[:, :, 0]) + posterior[:, :, 1] * torch.log(
            posterior[:, :, 1]
        )  # TODO check if this is correct

        regularization_loss += -torch.sum(entropy[torch.isfinite(entropy)])

        self.log("regularization_loss", regularization_loss)
        loss = reco_loss + self.beta * regularization_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
