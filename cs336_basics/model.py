import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding module: maps token IDs to embedding vectors.

    This is a custom implementation that matches PyTorch's nn.Embedding interface.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create parameter with shape (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Initialize using truncated normal distribution
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for given token IDs.

        Args:
            token_ids: Tensor of token IDs with arbitrary shape

        Returns:
            Tensor of embeddings with shape (*token_ids.shape, embedding_dim)
        """
        return self.weight[token_ids]


class Linear(nn.Module):
    """
    Linear transformation module: y = x @ W^T

    Note: We store W with shape (out_features, in_features) for memory ordering
    consistency with PyTorch's nn.Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create parameter W with shape (out_features, in_features)
        # This matches PyTorch's weight layout
        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Initialize using truncated normal distribution
        nn.init.trunc_normal_(self.W, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation: y = x @ W^T

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # x has shape (..., in_features)
        # W has shape (out_features, in_features)
        # We need output of shape (..., out_features)
        # So we compute: x @ W.T
        return x @ self.W.T
