"""PyTorch model architectures for multi-task intent prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionTransformerEncoder(nn.Module):
    """Transformer encoder over clickstream event sequences."""

    def __init__(
        self,
        event_vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.event_embedding = nn.Embedding(event_vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        event_ids: (batch, seq_len) int tensor
        mask: (batch, seq_len) bool tensor, True = padding
        """
        batch_size, seq_len = event_ids.shape
        positions = torch.arange(seq_len, device=event_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.event_embedding(event_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        # Mean pooling over non-padding positions
        if mask is not None:
            lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            x = x.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)

        return x


class MultiTaskIntentModel(nn.Module):
    """
    Multi-task neural network predicting:
    - intent class (4-way classification)
    - purchase probability (binary)
    - 30-day LTV (regression)
    - 7-day churn probability (binary)
    """

    def __init__(
        self,
        event_vocab_size: int,
        num_user_features: int = 32,
        num_product_features: int = 16,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        num_intent_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.sequence_encoder = SessionTransformerEncoder(
            event_vocab_size=event_vocab_size,
            embed_dim=embed_dim,
        )

        # Context feature projection
        self.user_proj = nn.Linear(num_user_features, 64)
        self.product_proj = nn.Linear(num_product_features, 32)

        # Shared representation
        shared_input_dim = embed_dim + 64 + 32
        self.shared = nn.Sequential(
            nn.Linear(shared_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.intent_classifier = nn.Linear(hidden_dim, num_intent_classes)
        self.purchase_head = nn.Linear(hidden_dim, 1)
        self.ltv_head = nn.Linear(hidden_dim, 1)
        self.churn_head = nn.Linear(hidden_dim, 1)

        # Learnable task uncertainty weights (Kendall et al.)
        self.log_sigma_intent = nn.Parameter(torch.zeros(1))
        self.log_sigma_purchase = nn.Parameter(torch.zeros(1))
        self.log_sigma_ltv = nn.Parameter(torch.zeros(1))
        self.log_sigma_churn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        event_ids: torch.Tensor,
        user_features: torch.Tensor,
        product_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_repr = self.sequence_encoder(event_ids, mask)
        user_repr = F.relu(self.user_proj(user_features))
        product_repr = F.relu(self.product_proj(product_features))

        combined = torch.cat([seq_repr, user_repr, product_repr], dim=-1)
        shared = self.shared(combined)

        return {
            "intent_logits": self.intent_classifier(shared),
            "purchase_logit": self.purchase_head(shared).squeeze(-1),
            "ltv": F.relu(self.ltv_head(shared)).squeeze(-1),
            "churn_logit": self.churn_head(shared).squeeze(-1),
            "shared_repr": shared,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Multi-task loss with uncertainty weighting.
        Lower log_sigma = higher weight for that task.
        """
        intent_loss = F.cross_entropy(outputs["intent_logits"], targets["intent"])
        purchase_loss = F.binary_cross_entropy_with_logits(
            outputs["purchase_logit"], targets["purchase"].float()
        )
        ltv_loss = F.mse_loss(outputs["ltv"], targets["ltv"])
        churn_loss = F.binary_cross_entropy_with_logits(
            outputs["churn_logit"], targets["churn"].float()
        )

        precision_intent = torch.exp(-self.log_sigma_intent)
        precision_purchase = torch.exp(-self.log_sigma_purchase)
        precision_ltv = torch.exp(-self.log_sigma_ltv)
        precision_churn = torch.exp(-self.log_sigma_churn)

        loss = (
            precision_intent * intent_loss + self.log_sigma_intent
            + precision_purchase * purchase_loss + self.log_sigma_purchase
            + precision_ltv * ltv_loss + self.log_sigma_ltv
            + precision_churn * churn_loss + self.log_sigma_churn
        )

        return loss / 4.0