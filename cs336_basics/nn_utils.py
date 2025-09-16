"""
Neural network utilities for CS336 Assignment 1.
This module contains utility functions for neural network operations.
"""

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


def get_batch(
    dataset: np.ndarray,
    batch_size: int, 
    context_length: int,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Desired batch size
        context_length: Context length for each example
        device: Device to place tensors on
        
    Returns:
        Tuple of (inputs, targets) tensors of shape (batch_size, context_length)
    """
    # Sample random starting positions
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input sequences and targets
    inputs = []
    targets = []
    
    for start_idx in start_indices:
        # Input is tokens at positions [start_idx : start_idx + context_length]
        input_seq = dataset[start_idx : start_idx + context_length]
        # Target is tokens at positions [start_idx + 1 : start_idx + context_length + 1]
        target_seq = dataset[start_idx + 1 : start_idx + context_length + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Convert to tensors
    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    
    return inputs, targets


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax along
        
    Returns:
        Softmax probabilities
    """
    return F.softmax(x, dim=dim)


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute average cross-entropy loss.
    
    Args:
        inputs: Logits of shape (batch_size, vocab_size)
        targets: Target class indices of shape (batch_size,)
        
    Returns:
        Average cross-entropy loss (scalar)
    """
    return F.cross_entropy(inputs, targets, reduction='mean')


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients to have maximum L2 norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm for gradients
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)