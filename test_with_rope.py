#!/usr/bin/env python3
"""
Test if the non-RoPE test actually expects RoPE
"""
import torch
import numpy as np
import json
from cs336_basics.model import scaled_dot_product_attention, rope

def test_with_rope_in_non_rope_test():
    """Test if adding RoPE to the non-RoPE test fixes it"""
    
    # Load expected output
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    # Load state dict
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # Create test input 
    torch.manual_seed(4)
    batch_size = 4
    n_queries = 12  
    d_model = 64
    num_heads = 4
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Testing multi-head attention WITH RoPE:")
    
    # Implementation WITH RoPE
    *batch_dims, seq_len, d_in = in_embeddings.shape
    d_k = d_model // num_heads
    theta = 10000.0  # From config
    
    # Apply projections
    q = torch.matmul(in_embeddings, q_proj_weight.T)
    k = torch.matmul(in_embeddings, k_proj_weight.T)
    v = torch.matmul(in_embeddings, v_proj_weight.T)
    
    # Reshape for multi-head attention
    q = q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    k = k.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    v = v.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    
    # Apply RoPE to queries and keys
    q_with_rope = rope(q, theta)
    k_with_rope = rope(k, theta)
    
    # Apply attention with RoPE
    attn_output = scaled_dot_product_attention(q_with_rope, k_with_rope, v)
    
    # Concatenate heads
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    
    # Final output projection
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    print(f"Actual first values: {output[0, 0, :5]}")
    print(f"Expected first values: {expected_output[0, 0, :5]}")
    
    # Compare
    diff = torch.abs(output - expected_output)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    # Test with pytest tolerance
    try:
        np.testing.assert_allclose(output.numpy(), expected_output.numpy(), rtol=1e-4, atol=1e-6)
        print("✅ Test PASSED with RoPE!")
        return True
    except AssertionError as e:
        print("❌ Test still FAILED with RoPE")
        return False

if __name__ == "__main__":
    test_with_rope_in_non_rope_test()