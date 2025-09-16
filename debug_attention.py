#!/usr/bin/env python3
"""
Debug multi-head attention implementation
"""
import torch
import numpy as np
from cs336_basics.model import scaled_dot_product_attention

def debug_multihead_attention():
    """Debug multi-head attention step by step"""
    
    # Load test data
    test_data = np.load('tests/_snapshots/test_multihead_self_attention.npz')
    expected_output = torch.from_numpy(test_data['array'])
    
    # Load state dict (same as conftest.py)
    import json
    
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights and move to CPU
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # Create test input (same as test fixture)
    batch_size = 4
    n_queries = 12  
    d_model = 64
    torch.manual_seed(4)
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Weight shapes:")
    print(f"Q: {q_proj_weight.shape}")
    print(f"K: {k_proj_weight.shape}")  
    print(f"V: {v_proj_weight.shape}")
    print(f"O: {o_proj_weight.shape}")
    print(f"Input: {in_embeddings.shape}")
    print(f"Expected output: {expected_output.shape}")
    
    # Manual implementation following adapter
    num_heads = 4
    d_k = d_model // num_heads  # 16
    
    batch_size, seq_len, _ = in_embeddings.shape
    
    # Apply projections
    q = torch.matmul(in_embeddings, q_proj_weight.T)  # (1, 16, 64)
    k = torch.matmul(in_embeddings, k_proj_weight.T)
    v = torch.matmul(in_embeddings, v_proj_weight.T)
    
    print(f"\nAfter projections:")
    print(f"Q: {q.shape}, range: [{q.min():.6f}, {q.max():.6f}]")
    print(f"K: {k.shape}, range: [{k.min():.6f}, {k.max():.6f}]")  
    print(f"V: {v.shape}, range: [{v.min():.6f}, {v.max():.6f}]")
    
    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, num_heads, d_k).transpose(-3, -2)  # (1, 4, 16, 16)
    k = k.view(batch_size, seq_len, num_heads, d_k).transpose(-3, -2)
    v = v.view(batch_size, seq_len, num_heads, d_k).transpose(-3, -2)
    
    print(f"\nAfter reshape:")
    print(f"Q: {q.shape}")
    print(f"K: {k.shape}")
    print(f"V: {v.shape}")
    
    # Apply attention
    attn_output = scaled_dot_product_attention(q, k, v)  # (1, 4, 16, 16)
    
    print(f"\nAfter attention:")
    print(f"Attention: {attn_output.shape}, range: [{attn_output.min():.6f}, {attn_output.max():.6f}]")
    
    # Concatenate heads
    attn_output = attn_output.transpose(-3, -2).contiguous().view(batch_size, seq_len, d_model)  # (1, 16, 64)
    
    print(f"\nAfter concatenation:")
    print(f"Concat: {attn_output.shape}, range: [{attn_output.min():.6f}, {attn_output.max():.6f}]")
    
    # Final projection
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    print(f"\nFinal output:")
    print(f"Output: {output.shape}, range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"Expected: {expected_output.shape}, range: [{expected_output.min():.6f}, {expected_output.max():.6f}]")
    
    # Compare
    diff = torch.abs(output - expected_output)
    print(f"\nDifference analysis:")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    # Check a few values
    print(f"\nSample comparisons:")
    for i in range(3):
        print(f"Position [{i}]: actual={output[0,0,i]:.6f}, expected={expected_output[0,0,i]:.6f}, diff={diff[0,0,i]:.6f}")

if __name__ == "__main__":
    debug_multihead_attention()