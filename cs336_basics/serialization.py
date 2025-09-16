"""
Model serialization utilities for CS336 Assignment 1.
This module contains functions for saving and loading model checkpoints.
"""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer, and iteration to a checkpoint file.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save  
        iteration: Current iteration number
        out: Output path or file-like object
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        src: Source path or file-like object  
        model: Model to restore state to
        optimizer: Optimizer to restore state to
        
    Returns:
        Iteration number from the checkpoint
    """
    checkpoint = torch.load(src, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    
    return iteration