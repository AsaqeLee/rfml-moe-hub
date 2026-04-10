"""Cross-attention multi-modal fusion modules."""

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer between two modalities.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Cross-attend from query modality to key_value modality.

        Args:
            query: (batch, embed_dim) query modality embedding.
            key_value: (batch, embed_dim) context modality embedding.

        Returns:
            Updated query embedding (batch, embed_dim).
        """
        # Add sequence dimension for MultiheadAttention: (batch, 1, embed_dim)
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)

        # Cross-attention with residual
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = self.norm1(q + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(1)  # (batch, embed_dim)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion across all modality pairs (bidirectional).

    Applies 2 layers of cross-attention between all i!=j modality pairs,
    then concatenates and projects to a single fused representation.

    Args:
        num_modalities: Number of input modalities (experts).
        embed_dim: Embedding dimension per modality.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_modalities: int = 4,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Cross-attention layers for each ordered pair (i, j) where i != j
        # Use ModuleDict keyed by "layer_i_j" for each layer
        self.cross_attn_layers = nn.ModuleDict()
        for layer in range(num_layers):
            for i in range(num_modalities):
                for j in range(num_modalities):
                    if i != j:
                        key = f"layer{layer}_q{i}_kv{j}"
                        self.cross_attn_layers[key] = CrossAttentionLayer(
                            embed_dim, num_heads, dropout
                        )

        # Project concatenated modalities to single representation
        self.projection = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality embeddings via cross-attention.

        Args:
            embeddings: List of tensors, each (batch, embed_dim).

        Returns:
            Fused representation (batch, embed_dim).
        """
        assert len(embeddings) == self.num_modalities

        current = list(embeddings)  # working copy

        for layer in range(self.num_layers):
            updated = []
            for i in range(self.num_modalities):
                # Aggregate cross-attention from all other modalities
                cross_outputs = []
                for j in range(self.num_modalities):
                    if i != j:
                        key = f"layer{layer}_q{i}_kv{j}"
                        cross_out = self.cross_attn_layers[key](current[i], current[j])
                        cross_outputs.append(cross_out)
                # Average cross-attention outputs from all other modalities
                updated_i = torch.stack(cross_outputs, dim=0).mean(dim=0)
                # Residual connection from input
                updated.append(current[i] + updated_i)
            current = updated

        # Concatenate all modality representations and project
        fused = torch.cat(current, dim=-1)  # (batch, num_modalities * embed_dim)
        return self.projection(fused)  # (batch, embed_dim)


class ConfidenceWeightedFusion(nn.Module):
    """Simpler baseline fusion using learned confidence weighting.

    Each expert produces a scalar confidence score; the fused output is
    a weighted average of embeddings. Typically 1-3% below learned
    cross-attention fusion.

    Args:
        num_modalities: Number of input modalities.
        embed_dim: Embedding dimension per modality.
    """

    def __init__(self, num_modalities: int = 4, embed_dim: int = 512):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        # One confidence head per modality
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            for _ in range(num_modalities)
        ])

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings via confidence-weighted average.

        Args:
            embeddings: List of tensors, each (batch, embed_dim).

        Returns:
            Fused representation (batch, embed_dim).
        """
        assert len(embeddings) == self.num_modalities

        # Compute confidence scores
        confidences = []
        for i, emb in enumerate(embeddings):
            conf = self.confidence_heads[i](emb)  # (batch, 1)
            confidences.append(conf)

        # Softmax over modalities for normalized weights
        conf_stack = torch.cat(confidences, dim=-1)  # (batch, num_modalities)
        weights = torch.softmax(conf_stack.float(), dim=-1)  # FP32
        weights = weights.to(embeddings[0].dtype)

        # Weighted average
        stacked = torch.stack(embeddings, dim=1)  # (batch, num_modalities, embed_dim)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)
        return fused
