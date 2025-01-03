import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Literal, Tuple

import numpy as np
import torch
from beartype import beartype
from lightning import Callback, LightningModule
from sklearn.metrics import adjusted_rand_score
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
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.is_beta_VAE = is_beta_VAE
        # self.beta = nn.Parameter(torch.tensor(beta), requires_grad=is_beta_VAE)
        self.beta = beta

    @beartype
    def forward(self, x: Tensor, target: Tensor) -> Dict:
        """Return the predicted mu_hat of the gaussian model of the likelihood.

        Args:
            data : Input data containing x and target with the shape (batch, *). #TODO verify dimensions

        Returns:
            any: Predictions.
        """
        # sigmoid activation to get the probability of the latent variable
        logits = self.encoder(x)  # unormalized log probabilities
        posterior = F.softmax(logits / self.temperature, dim=1)

        z_float = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        z_hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        # straight through gradient estimator
        z = (z_hard - z_float).detach() + z_float
        # z = torch.cat((1.0 - target.sum(dim= 1, keepdim=True), target), dim=1) #fake z for debug
        mu_hat = self.decoder(z)

        return {"mu_hat": mu_hat, "posterior": posterior, "z": z}

    @beartype
    def training_step(self, data: Any, batch_idx: int) -> Tensor:
        x, noise, mu, target = data
        dict = self.forward(x, target)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        z = dict["z"]
        reco_loss, reco_mean_loss, regularization_loss, loss = self.loss_function(x, mu, mu_hat, posterior)
        self.log("train_reconstruction_loss", reco_loss)
        self.log("train_mean_reconstruction_loss", reco_mean_loss)
        self.log("train_regularization_loss", regularization_loss)
        self.log("train_loss", loss)
        self.log("beta", self.beta)
        return loss

    @beartype
    def validation_step(self, data: Any, batch_idx: int) -> Tensor:
        x, noise, mu, target = data
        dict = self.forward(x, target)
        mu_hat = dict["mu_hat"]
        posterior = dict["posterior"]
        z = dict["z"]
        reco_loss, reco_mean_loss, regularization_loss, loss = self.loss_function(x, mu, mu_hat, posterior)
        self.log("val_reconstruction_loss", reco_loss)
        self.log("val_mean_reconstruction_loss", reco_mean_loss)
        self.log("val_regularization_loss", regularization_loss)
        self.log("val_loss", loss)

        # ARI calculation
        with torch.no_grad():
            target = torch.cat((1.0 - target.sum(dim=1, keepdim=True), target), dim=1)
            ari = adjusted_rand_score(target.argmax(dim=1).cpu().numpy(), z.argmax(dim=1).cpu().numpy())

        self.log("ARI", ari, prog_bar=True)
        return loss

    def loss_function(self, x, mu, mu_hat, posterior) -> Tensor:
        """

        Returns:
            Tensor: Losses (samples, tasks).
        """
        # elementwise squared error
        # MSE loss between the input and the output
        reco_loss = F.mse_loss(x, mu_hat, reduction="mean")
        with torch.no_grad():
            reco_mean_loss = F.mse_loss(x, mu, reduction="mean")
        observational_density = self.trainer.datamodule.train_dataset.observational_density
        prior = torch.full_like(posterior, fill_value=(1 - observational_density) / (posterior.shape[1] - 1))
        prior[:, 0] = observational_density
        regularization_loss = 0

        entropy = -torch.sum(posterior * torch.log(posterior), dim=1) + torch.sum(
            posterior * torch.log(prior), dim=1
        )

        regularization_loss -= torch.mean(entropy[torch.isfinite(entropy)])

        loss = reco_loss + self.beta * regularization_loss

        return reco_loss, reco_mean_loss, regularization_loss, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CyclicalAnnealingCallback(Callback):
    def __init__(self, max_value, min_value, period, max_cycle, warmup_epochs=0):
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.period = period
        self.max_cycle = max_cycle
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch < self.warmup_epochs:
            new_beta = self.min_value
            current_cycle = 0
        else:
            adjusted_epoch = current_epoch - self.warmup_epochs
            current_period_epoch = adjusted_epoch % self.period
            current_cycle = adjusted_epoch // self.period
            scale_factor = min(1, current_period_epoch / (self.period / 2))
            new_beta = self.min_value + (self.max_value - self.min_value) * scale_factor

        if current_cycle < self.max_cycle:
            setattr(pl_module, "beta", new_beta)
