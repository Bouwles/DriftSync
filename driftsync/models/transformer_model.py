"""
Transformer Encoder-Based Drift Predictor
==========================================
Causal (masked) multi-head self-attention encoder stack for temporal
sequence modelling.

Architecture
------------
    Input (B, L, F)
        ↓
    Input Projection  (Linear -> LayerNorm)
        ↓
    Sinusoidal Positional Encoding
        ↓
    ┌─────────────────────────────────────┐
    │  TransformerEncoderLayer × N        │
    │  (MultiHeadSelfAttention            │
    │   + Pre-LN + FFN + Dropout)         │
    └─────────────────────────────────────┘
        ↓
    Global Average Pooling over sequence
        ↓
    Classification Head (Linear -> GELU -> Dropout -> Linear)
        ↓
    Logit (B,)

Notes
-----
- Pre-LayerNorm (vs. post-LN) for training stability.
- Causal mask ensures no look-ahead during training/inference.
- Attention weights are stored after last forward pass for visualisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from driftsync.models.base import DriftPredictor
from driftsync.configs import TransformerConfig


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Pre-LN Transformer Encoder Layer
# ---------------------------------------------------------------------------

class PreLNTransformerLayer(nn.Module):
    """
    Pre-LayerNorm Transformer encoder layer.

    Pre-LN applies layer norm before each sub-block (attention / FFN)
    rather than after, leading to more stable gradients.

    Sub-blocks:
        h = x + Attention(LayerNorm(x))
        out = h + FFN(LayerNorm(h))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Store last attention weights for visualisation
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- Self-Attention block ---
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attn_weights = attn_weights.detach()
        x = x + attn_out

        # --- FFN block ---
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Drift Predictor
# ---------------------------------------------------------------------------

class TransformerDriftPredictor(DriftPredictor):
    """
    Transformer encoder for cognitive drift prediction.

    Args:
        cfg: TransformerConfig instance.
    """

    def __init__(self, cfg: TransformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()

        d = self.cfg.d_model

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(self.cfg.input_dim, d),
            nn.LayerNorm(d),
        )

        # --- Positional encoding ---
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=d,
            max_len=self.cfg.max_seq_len,
            dropout=self.cfg.dropout,
        )

        # --- Encoder stack ---
        self.encoder_layers = nn.ModuleList([
            PreLNTransformerLayer(
                d_model=d,
                nhead=self.cfg.nhead,
                dim_feedforward=self.cfg.dim_feedforward,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.num_encoder_layers)
        ])

        self.final_norm = nn.LayerNorm(d)

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d // 2, self.cfg.output_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier / Glorot uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Causal mask
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular causal mask.
        Positions where mask=True are IGNORED by MultiheadAttention.

        mask[i, j] = True means position i cannot attend to position j.
        For a causal model: j > i (future) is masked.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

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
        B, L, _ = x.shape

        # Project + positional encode
        out = self.input_proj(x)   # (B, L, d_model)
        out = self.pos_enc(out)

        # Causal attention mask
        mask = self._causal_mask(L, x.device)

        # Encoder stack
        for layer in self.encoder_layers:
            out = layer(out, attn_mask=mask)

        out = self.final_norm(out)

        # Global average pooling over the sequence dimension
        pooled = out.mean(dim=1)   # (B, d_model)

        logits = self.head(pooled).squeeze(-1)   # (B,)
        return logits

    # ------------------------------------------------------------------
    # Attention visualisation
    # ------------------------------------------------------------------

    def get_attention_maps(self) -> list[torch.Tensor]:
        """
        Return attention weight tensors from all encoder layers.

        Each tensor has shape (B, L, L) — the last batch processed.
        Only valid after a forward() call.
        """
        return [
            layer.last_attn_weights
            for layer in self.encoder_layers
            if layer.last_attn_weights is not None
        ]
