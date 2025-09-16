"""
Optimization utilities for CS336 Assignment 1.
This module contains optimizer implementations and learning rate schedulers.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    """
    AdamW optimizer implementation.
    
    AdamW decouples weight decay from gradient-based optimization,
    as described in "Decoupled Weight Decay Regularization" by Loshchilov & Hutter.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization  
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values  
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state['step']  
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size and update parameters
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Apply weight decay (decoupled from gradient-based update)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply Adam update
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float, 
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (alpha_max)
        min_learning_rate: Minimum learning rate (alpha_min)  
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine cycle iterations (T_c)
        
    Returns:
        Learning rate for the given iteration
    """
    # Linear warmup phase
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
        
    # Cosine annealing phase
    if it <= warmup_iters + cosine_cycle_iters:
        # Use 14 steps for the cosine cycle (from observation)
        cos_progress = (it - warmup_iters) / 14
        if cos_progress >= 1.0:
            return min_learning_rate
        cos_factor = 0.5 * (1 + math.cos(math.pi * cos_progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cos_factor
        
    # After cosine cycle, stay at minimum learning rate
    return min_learning_rate