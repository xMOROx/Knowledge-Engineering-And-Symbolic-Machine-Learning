import random
import torch
from typing import List


class ExperienceMemory:
    """
    A simple circular buffer for storing experience transitions.
    Uses a potentially custom replacement strategy.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initializes the ExperienceMemory.

        Args:
            capacity: The maximum number of transitions to store.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.memory: List[torch.Tensor] = []
        self.pos = 0
        # Indices derived from the packet format structure in server.py
        # If state_dims=8, reward is index 9, terminal is index 18
        # state(8) + action(1) + reward(1) + next_state(8) + terminal(1) = 19 elements
        # Index 9 = reward | Index 18 = terminal
        # Original code used index 9 for check. Let's assume state_dims=8 implicitly here.
        # TODO: Make indices dependent on state_dims passed to server?
        self._reward_idx = 8 + 1  # Assumes action is 1 element
        self._terminal_idx = (
            self._reward_idx + 1 + 8
        )  # Assumes next_state is 8 elements

    def record_transition(self, transition: torch.Tensor) -> None:
        """
        Adds a transition to the memory. Overwrites older transitions
        using a specific strategy if capacity is reached.

        Args:
            transition: A tensor representing the transition, expected shape (1, features).
                        The tensor should contain (state, action, reward, next_state, terminal).
        """

        if transition.dim() == 2 and transition.shape[0] == 1:
            transition_squeezed = transition.squeeze(0)
        elif transition.dim() == 1:
            transition_squeezed = transition
        else:
            raise ValueError(
                f"Unexpected transition shape: {transition.shape}. Expected (1, features) or (features,)."
            )

        if self._terminal_idx >= len(transition_squeezed):
            raise IndexError(
                f"Calculated terminal index ({self._terminal_idx}) is out of bounds "
                f"for transition length ({len(transition_squeezed)}). "
                f"Check state_dims consistency."
            )

        if len(self.memory) < self.capacity:
            self.memory.append(transition_squeezed)
        else:
            # --- Custom Replacement Logic ---
            # Original logic: Preferentially overwrite non-terminal states
            # with low rewards (based on index 9, assumed to be reward).
            # Keep terminal states with 90% probability.
            # TODO: Verify if this custom logic is intended or if standard FIFO/random is better.
            is_terminal = self.memory[self.pos][self._terminal_idx] > 0

            # Keep terminal states with high probability, otherwise replace
            if is_terminal and random.random() < 0.9:
                self.pos = (self.pos + 1) % self.capacity
                num_checked = 0
                while (
                    self.memory[self.pos][self._terminal_idx] > 0
                    and random.random() < 0.9
                ):
                    self.pos = (self.pos + 1) % self.capacity
                    num_checked += 1
                    if num_checked >= self.capacity:
                        break

            self.memory[self.pos] = transition_squeezed
            self.pos = (self.pos + 1) % self.capacity

    def get_batch(self, batch_size: int = 32) -> torch.Tensor:
        """
        Samples a random batch of transitions from the memory.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            A tensor containing the batch of transitions, shape (batch_size, features).

        Raises:
            ValueError: If batch_size is larger than the number of stored transitions.
        """
        if batch_size > len(self.memory):
            raise ValueError(
                f"Requested batch size {batch_size} is larger than memory size {len(self.memory)}"
            )

        sampled_transitions = random.sample(self.memory, batch_size)

        return torch.stack(sampled_transitions)

    def __len__(self) -> int:
        """Returns the current number of transitions stored in the memory."""
        return len(self.memory)
