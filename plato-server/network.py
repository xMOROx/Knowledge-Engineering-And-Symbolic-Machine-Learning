import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Simple Feed-Forward Neural Network for Q-value estimation.
    Architecture: Input -> FC -> ReLU -> FC -> ReLU -> FC -> Output
    """

    def __init__(
        self, state_dims: int = 5, action_dims: int = 6, hidden_dims: int = 128
    ):
        """
        Initializes the QNetwork.

        Args:
            state_dims: The number of dimensions in the input state.
            action_dims: The number of possible actions (output size).
            hidden_dims: The number of units in the hidden layers.
        """
        super().__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(state_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.out = nn.Linear(hidden_dims, action_dims)

        logging.info(
            f"Initialized QNetwork: state_dims={state_dims}, "
            f"action_dims={action_dims}, hidden_dims={hidden_dims}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x: The input state tensor of shape (batch_size, state_dims).

        Returns:
            A tensor of shape (batch_size, action_dims) representing Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values
