"""
Neural network model components for CS336 Assignment 1.
This module contains implementations of various neural network layers and components
used in Transformer architectures.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    """
    A linear (fully connected) layer implementation.
    """
    
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        """
        Initialize a linear layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension  
            bias: Whether to include bias term
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * math.sqrt(2.0 / d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Forward pass through the linear layer.
        
        Args:
            x: Input tensor of shape (..., d_in)
            
        Returns:
            Output tensor of shape (..., d_out)
        """
        output = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        return output


class Embedding(nn.Module):
    """
    Token embedding layer.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding size)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * math.sqrt(1.0 / d_model))
        
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """
        Get embeddings for token IDs.
        
        Args:
            token_ids: Token IDs to embed
            
        Returns:
            Embeddings for the tokens
        """
        return self.weight[token_ids]


class SwiGLU(nn.Module):
    """
    SwiGLU activation function implementation.
    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
    Where Swish(x) = x * sigmoid(x) = x * σ(x)
    """
    
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        """
        Initialize SwiGLU layer.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            bias: Whether to use bias
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=bias)  # Gate projection
        self.w2 = Linear(d_ff, d_model, bias=bias)  # Down projection 
        self.w3 = Linear(d_model, d_ff, bias=bias)  # Up projection
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Forward pass through SwiGLU.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # SwiGLU formula: W2(Swish(W1(x)) ⊙ W3(x))
        gate = self.w1(x)  # Gate
        up = self.w3(x)    # Up projection
        swish_gate = gate * torch.sigmoid(gate)  # Swish activation
        return self.w2(swish_gate * up)  # Down projection


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"], 
    V: Float[Tensor, "... values d_v"],
    mask: Optional[Bool[Tensor, "... queries keys"]] = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Scaled dot-product attention implementation.
    
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor  
        mask: Optional attention mask
        
    Returns:
        Attention output
    """
    d_k = Q.size(-1)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Convert boolean mask: True means keep, False means mask (set to -inf)
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply weights to values
    output = torch.matmul(attn_weights, V)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Separate QKV projections to match test key names
        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.output_proj = Linear(d_model, d_model, bias=bias)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention output  
        """
        *batch_dims, seq_len, d_model = x.shape
        
        # Project to Q, K, V separately
        q = self.q_proj(x)  # (..., seq_len, d_model)
        k = self.k_proj(x)  # (..., seq_len, d_model)
        v = self.v_proj(x)  # (..., seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        # Apply scaled dot-product attention  
        attn_output = scaled_dot_product_attention(q, k, v)  # (..., num_heads, seq_len, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(-3, -2)  # (..., seq_len, num_heads, d_k)
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len, d_model)
        
        # Final output projection
        output = self.output_proj(attn_output)
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Forward pass through RMSNorm.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.weight * (x / rms)


def rope(
    x: Float[Tensor, "... seq_len d_k"],
    theta: float = 10000.0,
) -> Float[Tensor, "... seq_len d_k"]:
    """
    Rotary Position Embedding (RoPE) implementation.
    
    Args:
        x: Input tensor
        theta: Base for the rotation frequencies
        
    Returns:
        Tensor with RoPE applied
    """
    *batch_dims, seq_len, d_k = x.shape
    device = x.device
    
    # Generate position indices
    pos = torch.arange(seq_len, device=device).float()
    
    # Generate frequency dimension indices  
    freqs = torch.arange(0, d_k, 2, device=device).float()
    
    # Compute frequencies: theta^(-2i/d_k) for i in [0, d_k/2)
    freqs = 1.0 / (theta ** (freqs / d_k))
    
    # Compute position encodings: pos * freq for each (pos, freq) pair
    pos_enc = torch.outer(pos, freqs)  # (seq_len, d_k/2)
    
    # Create cos and sin components
    cos_enc = torch.cos(pos_enc)  # (seq_len, d_k/2)
    sin_enc = torch.sin(pos_enc)  # (seq_len, d_k/2)
    
    # Split x into even and odd dimensions
    x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
    x_odd = x[..., 1::2]   # (..., seq_len, d_k/2)
    
    # Apply rotation
    x_even_rot = x_even * cos_enc - x_odd * sin_enc
    x_odd_rot = x_even * sin_enc + x_odd * cos_enc
    
    # Interleave back  
    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (..., seq_len, d_k/2, 2)
    x_rot = x_rot.flatten(-2)  # (..., seq_len, d_k)
    
    return x_rot


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embedding.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0, bias: bool = False):
        """
        Initialize multi-head attention with RoPE.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            theta: RoPE theta parameter
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.theta = theta
        
        # QKV projections - match test key names
        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)  
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.output_proj = Linear(d_model, d_model, bias=bias)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass through multi-head attention with RoPE.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention output
        """
        *batch_dims, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (..., seq_len, d_model)
        k = self.k_proj(x)  # (..., seq_len, d_model)  
        v = self.v_proj(x)  # (..., seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)  
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        # Now: (..., num_heads, seq_len, d_k)
        
        # Apply RoPE to queries and keys
        q = rope(q, self.theta)
        k = rope(k, self.theta)
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(q, k, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(-3, -2)  # (..., seq_len, num_heads, d_k)
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len, d_model)
        
        # Final output projection
        output = self.output_proj(attn_output)
        
        return output


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    SiLU (Swish) activation function: x * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        SiLU activated tensor
    """
    return x * torch.sigmoid(x)


class PositionwiseFeedforward(nn.Module):
    """
    Position-wise feedforward network using SwiGLU activation.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize feedforward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feedforward hidden dimension
        """
        super().__init__()
        # Use test key names directly
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False) 
        self.w3 = Linear(d_model, d_ff, bias=False)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass through feedforward network using SwiGLU.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # SwiGLU: silu(W1 @ x) * (W3 @ x) then W2 @ result
        return self.w2(silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    A single Transformer block with attention and feedforward layers.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feedforward hidden dimension  
            max_seq_len: Maximum sequence length
            use_rope: Whether to use RoPE
            rope_theta: RoPE theta parameter
        """
        super().__init__()
        
        # Use test key names
        self.ln1 = RMSNorm(d_model)
        
        if use_rope:
            self.attn = MultiHeadSelfAttentionWithRoPE(
                d_model, num_heads, max_seq_len, rope_theta
            )
        else:
            self.attn = MultiHeadSelfAttention(d_model, num_heads)
            
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass through Transformer block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Pre-norm attention with residual connection
        x = x + self.attn(self.ln1(x))
        
        # Pre-norm feedforward with residual connection  
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            d_ff: Feedforward hidden dimension
            max_seq_len: Maximum sequence length
            use_rope: Whether to use RoPE
            rope_theta: RoPE theta parameter  
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Token embedding - use test key name
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                use_rope=use_rope,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm - use test key name
        self.ln_final = RMSNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        
        # Tie embedding and output weights (optional but common)
        self.lm_head.weight = self.token_embeddings.weight
        
    def forward(
        self,
        input_ids: Int[Tensor, "... seq_len"],
        targets: Optional[Int[Tensor, "... seq_len"]] = None,
    ) -> tuple[Float[Tensor, "... seq_len vocab_size"], Optional[Float[Tensor, ""]]]:
        """
        Forward pass through Transformer LM.
        
        Args:
            input_ids: Input token IDs
            targets: Target token IDs (for loss calculation)
            
        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
        """
        # Get embeddings
        x = self.token_embeddings(input_ids)  # (..., seq_len, d_model)
        
        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Final norm  
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (..., seq_len, vocab_size)
        
        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            # Reshape for loss computation: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            
        return logits, loss
        
    def generate(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Int[Tensor, "batch_size new_seq_len"]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token sequence
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure we don't exceed max sequence length
        if seq_len >= self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len + 1:]
            
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # Use only the last max_seq_len tokens
                input_chunk = generated[:, -self.max_seq_len:]
                logits, _ = self.forward(input_chunk)
                
                # Get logits for the last position
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Sample from the distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
        return generated