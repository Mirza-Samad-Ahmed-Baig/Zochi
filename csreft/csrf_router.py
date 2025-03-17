import torch
import torch.nn as nn
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class CSRFRouter(nn.Module):
    """
    CSRF Router module that gates which subspace modules to activate for each incoming query.

    This module analyzes incoming queries and activates the relevant subspaces at inference time.
    """

    def __init__(
        self,
        input_dim: int,
        num_subspaces: int,
        hidden_dim: Optional[int] = None,
        activation: str = 'relu',
        gating_threshold: float = 0.5,
    ):
        """
        Initializes the CSRFRouter.

        Args:
            input_dim (int): Dimension of the input embeddings.
            num_subspaces (int): Number of subspace modules available.
            hidden_dim (Optional[int]): Dimension of the hidden layer. Defaults to input_dim // 2.
            activation (str): Activation function to use. One of 'relu', 'tanh', 'gelu'. Defaults to 'relu'.
            gating_threshold (float): Threshold for binarizing gating probabilities. Defaults to 0.5.
        """
        super(CSRFRouter, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2  # Default hidden dimension

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation_fn(activation)
        self.fc2 = nn.Linear(hidden_dim, num_subspaces)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)   # <-- small random bias
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        
        # For gating outputs, we can use sigmoid for independent gating
        self.gate_fn = nn.Sigmoid()
        self.gating_threshold = gating_threshold

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the router.

        Args:
            input_embeddings (torch.Tensor): Tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Gating probabilities for each subspace module (batch_size, num_subspaces)
        """
        x = input_embeddings.float()
        x = torch.clamp(x, -10, 10)
        x = self.fc1(x)
        x = self.activation(x)
        x = torch.clamp(x, -10, 10)
        logits = self.fc2(x)
        logits = torch.clamp(logits, -10, 10)
        gating_probs = self.gate_fn(logits)
        if torch.isnan(gating_probs).any():
            logger.warning(f"[CSRFRouter] gating_probs has NaNs! logits range=({logits.min().item()}, {logits.max().item()})")

            logger.debug(
                f"[CSRFRouter] gating_probs shape={gating_probs.shape}, "
                f"mean={gating_probs.mean().item():.4f}, min={gating_probs.min().item():.4f}, "
                f"max={gating_probs.max().item():.4f}"
            )
        return gating_probs

    def get_active_subspaces(self, gating_probs: torch.Tensor) -> torch.Tensor:
        """
        Get the active subspaces based on the gating probabilities and threshold.

        Args:
            gating_probs (torch.Tensor): Gating probabilities from forward pass (batch_size, num_subspaces)

        Returns:
            torch.Tensor: Binary tensor indicating active subspaces (batch_size, num_subspaces)
        """
        return gating_probs

    @staticmethod
    def _get_activation_fn(activation: str) -> Callable:
        """
        Returns the activation function corresponding to the provided activation string.

        Args:
            activation (str): Name of the activation function.

        Returns:
            Callable: Activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

