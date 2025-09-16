#!/usr/bin/env python3
"""
Let's try to work backwards from the expected output to understand what went wrong
"""
import torch
import numpy as np
import json
from cs336_basics.model import scaled_dot_product_attention

def reverse_engineer_multihead_attention():
    """Try to figure out the issue by working backwards"""
    
    # Load expected output and state dict
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # Create test input 
    batch_size = 4
    n_queries = 12  
    d_model = 64
    num_heads = 4
    torch.manual_seed(4)
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Debugging step by step...")
    print(f"Input shape: {in_embeddings.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    
    # Step 1: Check if the issue might be in how I'm accessing the test
    # Let me run the actual pytest to see what it says about the difference
    from tests.adapters import run_multihead_self_attention
    
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    
    print(f"Actual output first few values: {actual_output[0, 0, :5]}")
    print(f"Expected output first few values: {expected_output[0, 0, :5]}")
    
    # Let me try a different seed or check if there's randomness elsewhere
    print("\nTrying different random seed...")
    torch.manual_seed(42)  # Different seed
    in_embeddings_42 = torch.randn(batch_size, n_queries, d_model)
    
    actual_output_42 = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings_42,
    )
    
    print(f"Output with seed 42: {actual_output_42[0, 0, :5]}")
    
    # Maybe the issue is that our scaled_dot_product_attention has a bug?
    # Let me check with a simple manual implementation
    print("\nTesting manual attention computation...")
    
    *batch_dims, seq_len, d_in = in_embeddings.shape
    d_k = d_model // num_heads
    
    # Apply projections
    q = torch.matmul(in_embeddings, q_proj_weight.T)
    k = torch.matmul(in_embeddings, k_proj_weight.T) 
    v = torch.matmul(in_embeddings, v_proj_weight.T)
    
    # Reshape for heads
    q = q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    k = k.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    v = v.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    
    # Manual attention computation (to double-check our attention function)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output_manual = torch.matmul(attn_weights, v)
    
    # Using our function
    attn_output_func = scaled_dot_product_attention(q, k, v)
    
    print(f"Manual vs function attention diff: {torch.abs(attn_output_manual - attn_output_func).max():.6f}")
    
    # Continue with concatenation and output projection
    attn_output_manual = attn_output_manual.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    output_manual = torch.matmul(attn_output_manual, o_proj_weight.T)
    
    print(f"Manual implementation output: {output_manual[0, 0, :5]}")
    print(f"Adapter implementation output: {actual_output[0, 0, :5]}")
    print(f"Manual vs adapter diff: {torch.abs(output_manual - actual_output).max():.6f}")
    
if __name__ == "__main__":
    reverse_engineer_multihead_attention()