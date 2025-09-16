#!/usr/bin/env python3
"""
Test different interpretations of the multi-head attention adapter
"""
import torch
import numpy as np
import json
from cs336_basics.model import scaled_dot_product_attention

def test_interpretation_1():
    """
    Interpretation 1: d_k means d_model in the type annotation
    The weight matrices are [d_model, d_model] for complete projections
    """
    print("=== Interpretation 1: d_k = d_model ===")
    
    # Load expected output
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    # Load state dict
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights (move to CPU)
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
    
    # Implementation where d_k = d_model (current implementation)
    *batch_dims, seq_len, d_in = in_embeddings.shape
    d_k_for_heads = d_model // num_heads  # This is 16
    
    # Apply projections to get full d_model outputs
    q = torch.matmul(in_embeddings, q_proj_weight.T)  # (..., seq_len, d_model)
    k = torch.matmul(in_embeddings, k_proj_weight.T)
    v = torch.matmul(in_embeddings, v_proj_weight.T)
    
    # Reshape for multi-head attention: split d_model across heads
    q = q.view(*batch_dims, seq_len, num_heads, d_k_for_heads).transpose(-3, -2)
    k = k.view(*batch_dims, seq_len, num_heads, d_k_for_heads).transpose(-3, -2) 
    v = v.view(*batch_dims, seq_len, num_heads, d_k_for_heads).transpose(-3, -2)
    
    # Apply scaled dot-product attention
    attn_output = scaled_dot_product_attention(q, k, v)  
    
    # Concatenate heads
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    
    # Final output projection
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    # Compare
    diff = torch.abs(output - expected_output)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    return output


def test_interpretation_2():
    """
    Interpretation 2: The weights need different processing
    Maybe they're organized differently
    """
    print("\n=== Interpretation 2: Different organization ===")
    
    # Load expected output
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    # Load state dict
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights (move to CPU)
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
    
    # Try treating each weight as separate head weights
    # Maybe the weights are [d_model, d_model] but should be treated as 
    # [num_heads * d_k, d_model] where d_k = d_model // num_heads = 16
    
    *batch_dims, seq_len, d_in = in_embeddings.shape
    d_k = d_model // num_heads  # 16
    
    # Apply projections
    q = torch.matmul(in_embeddings, q_proj_weight.T)  # (..., seq_len, 64)
    k = torch.matmul(in_embeddings, k_proj_weight.T)  # (..., seq_len, 64)
    v = torch.matmul(in_embeddings, v_proj_weight.T)  # (..., seq_len, 64)
    
    # Reshape differently: treat the 64 outputs as 4 heads of 16 dimensions each
    # But this is the same as interpretation 1...
    
    # Let me try a different approach: what if the attention computation is different?
    # Let's check if there's a scaling factor or different order
    
    q = q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    k = k.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2) 
    v = v.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    
    # Try manual attention with different scaling
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    
    # Concatenate heads
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
    
    # Final output projection
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    # Compare
    diff = torch.abs(output - expected_output)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    return output

if __name__ == "__main__":
    out1 = test_interpretation_1()
    out2 = test_interpretation_2()
    
    print(f"\nComparison between interpretations:")
    diff_between = torch.abs(out1 - out2)
    print(f"Max diff between methods: {diff_between.max():.6f}")