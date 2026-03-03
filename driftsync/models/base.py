"""
Base Model Interface
====================
Abstract base class that both LSTM and Transformer models inherit from.
Provides:
  - A unified forward(x) -> logit interface.
  - Monte Carlo Dropout inference for uncertainty estimation.
  - Parameter counting utility.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class DriftPredictor(ABC, nn.Module):
    """
    Abstract base for all DriftSync sequence models.

    Subclasses must implement `forward(x) -> Tensor` which returns raw
    logits (pre-sigmoid) of shape (batch,).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch,) — unbounded scalar per sample.
        """
        ...

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid of logits -> probability in [0, 1].

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            probabilities: (batch,)
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def mc_dropout_predict(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Keeps dropout active during inference by calling model.train()
        temporarily, then aggregates N stochastic forward passes.

        Returns:
            mean_proba:  (batch,) — mean predicted probability.
            uncertainty: (batch,) — standard deviation across samples.

        Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016).
        """
        # Enable dropout by switching to training mode temporarily
        training_before = self.training
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                samples.append(torch.sigmoid(logits))

        # Restore original mode
        if not training_before:
            self.eval()

        samples_tensor = torch.stack(samples, dim=0)   # (n_samples, batch)
        mean_proba  = samples_tensor.mean(dim=0)        # (batch,)
        uncertainty = samples_tensor.std(dim=0)         # (batch,)
        return mean_proba, uncertainty

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self) -> None:
        """Re-initialise all weights (calls _init_weights if defined)."""
        for module in self.modules():
            if hasattr(module, "_init_weights"):
                module._init_weights()
