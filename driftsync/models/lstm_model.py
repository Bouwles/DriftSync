"""
LSTM-Based Drift Predictor
===========================
Multi-layer LSTM with:
  - Configurable depth and hidden dimension.
  - Layer normalisation between LSTM layers.
  - Dropout on all inter-layer connections.
  - Orthogonal weight initialisation for LSTM gates.
  - Optional bidirectional encoding.

Architecture
------------
    Input (B, L, F)
        ↓
    InputProjection (Linear + LayerNorm)
        ↓
    LSTM Layer 1  -> LayerNorm -> Dropout
        ↓
    LSTM Layer 2  -> LayerNorm -> Dropout
        ↓
    ...
        ↓
    Last hidden state h_T  (B, H)
        ↓
    Classification Head (Linear -> Dropout -> Linear)
        ↓
    Logit (B,)
"""

from __future__ import annotations


import torch
import torch.nn as nn
from driftsync.models.base import DriftPredictor
from driftsync.configs import LSTMConfig


class LSTMDriftPredictor(DriftPredictor):
    """
    Stacked LSTM for cognitive drift prediction.

    Args:
        cfg: LSTMConfig instance.
    """

    def __init__(self, cfg: LSTMConfig | None = None):
        super().__init__()
        self.cfg = cfg or LSTMConfig()

        D = self.cfg.hidden_dim
        dirs = 2 if self.cfg.bidirectional else 1

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(self.cfg.input_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )

        # --- LSTM stack (one cell per layer for fine-grained control) ---
        self.lstm_cells: nn.ModuleList = nn.ModuleList()
        self.layer_norms: nn.ModuleList = nn.ModuleList()
        self.dropouts: nn.ModuleList = nn.ModuleList()

        for layer_idx in range(self.cfg.num_layers):
            in_dim = D * dirs if layer_idx == 0 else D * dirs
            cell = nn.LSTM(
                input_size=in_dim,
                hidden_size=D,
                num_layers=1,
                batch_first=True,
                bidirectional=self.cfg.bidirectional,
            )
            self.lstm_cells.append(cell)
            self.layer_norms.append(nn.LayerNorm(D * dirs))
            self.dropouts.append(nn.Dropout(self.cfg.dropout))

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Linear(D * dirs, D // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(D // 2, self.cfg.output_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Orthogonal initialisation for LSTM weights (prevents vanishing gradients).
        Zeros for biases, except forget gate bias set to 1.0.
        """
        for cell in self.lstm_cells:
            for name, param in cell.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
                    # Set forget gate bias to 1 for better gradient flow
                    hidden_size = param.data.shape[0] // 4
                    param.data[hidden_size: 2 * hidden_size].fill_(1.0)

        # Head
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch,)
        """
        # Project input features
        out = self.input_proj(x)   # (B, L, D)

        # Stack LSTM layers with residual connections where dimensions match
        prev_dim = out.shape[-1]
        for cell, ln, drop in zip(self.lstm_cells, self.layer_norms, self.dropouts):
            lstm_out, _ = cell(out)          # (B, L, D*dirs)
            lstm_out = ln(lstm_out)
            lstm_out = drop(lstm_out)
            # Residual connection when dimensions match
            if lstm_out.shape[-1] == out.shape[-1]:
                out = lstm_out + out
            else:
                out = lstm_out

        # Take the last valid timestep
        h_last = out[:, -1, :]              # (B, D*dirs)

        logits = self.head(h_last).squeeze(-1)   # (B,)
        return logits
