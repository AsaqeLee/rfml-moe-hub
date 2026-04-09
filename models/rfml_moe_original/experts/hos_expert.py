"""HOS-domain Expert: FT-Transformer for tabular higher-order statistics.

Processes a vector of HOS features through learned per-feature embeddings
(feature tokenizer) and a transformer encoder to produce 512-dim embeddings.
"""

import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Tokenize each numerical feature into a learned embedding vector.

    Each of the `num_features` scalar inputs gets its own linear projection
    into `d_model` dimensions, producing a sequence of feature tokens.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        # Each feature has its own weight vector and bias
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize features.

        Args:
            x: (batch, num_features)

        Returns:
            (batch, num_features, d_model) — one token per feature.
        """
        # x: (batch, num_features) -> (batch, num_features, 1)
        # weight: (num_features, d_model)
        # result: (batch, num_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class HOSExpert(nn.Module):
    """FT-Transformer for higher-order statistics features.

    Input:  (batch, num_features) where num_features=20
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Architecture:
        Feature tokenizer (per-feature linear embeddings) + [CLS] token
        -> 3-layer transformer encoder (8 heads, 192-dim)
        -> MLP projection to 512-dim embedding

    Parameter count: ~2-5M
    """

    def __init__(
        self,
        num_features: int = 20,
        num_classes: int = 10,
        d_model: int = 192,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 768,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model

        # Feature tokenizer: each feature -> d_model-dim token
        self.tokenizer = FeatureTokenizer(num_features, d_model)

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional embeddings for num_features + 1 (CLS) tokens
        self.pos_embedding = nn.Parameter(torch.empty(1, num_features + 1, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Projection to 512-dim embedding
        self.projection = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run tokenizer + transformer, return 512-dim embedding."""
        batch_size = x.size(0)

        # Tokenize features: (batch, num_features, d_model)
        tokens = self.tokenizer(x)

        # Prepend [CLS] token: (batch, 1 + num_features, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embedding

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Extract [CLS] output
        cls_out = tokens[:, 0]  # (batch, d_model)

        # Project to embed_dim
        return self.projection(cls_out)  # (batch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: HOS feature vector of shape (batch, num_features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: HOS feature vector of shape (batch, num_features).

        Returns:
            Embedding of shape (batch, 512).
        """
        return self._extract_features(x)

    def freeze(self):
        """Freeze all parameters for progressive training."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
