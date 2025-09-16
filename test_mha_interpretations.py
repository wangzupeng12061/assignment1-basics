#!/usr/bin/env python3
"""
Try different multi-head attention implementation interpretations
"""
import torch
import numpy as np
import json
from cs336_basics.model import scaled_dot_product_attention

def test_different_mha_interpretations():
    """Test different ways to interpret multi-head attention"""
    
    # Load expected and setup
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    torch.manual_seed(4)
    batch_size, n_queries, d_model = 4, 12, 64
    num_heads = 4
    d_k = d_model // num_heads  # 16
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Testing different multi-head attention interpretations...")
    
    # Current implementation (batch projection then split)
    def current_implementation():
        q = torch.matmul(in_embeddings, q_proj_weight.T)  # (4, 12, 64)
        k = torch.matmul(in_embeddings, k_proj_weight.T)
        v = torch.matmul(in_embeddings, v_proj_weight.T)
        
        q = q.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)  # (4, 4, 12, 16)
        k = k.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
        v = v.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
        
        attn_output = scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(batch_size, n_queries, d_model)
        output = torch.matmul(attn_output, o_proj_weight.T)
        return output
    
    # Alternative: Split weights by heads first
    def split_weights_implementation():
        # Split the weights into per-head weights
        q_heads = q_proj_weight.view(num_heads, d_k, d_model)  # (4, 16, 64)
        k_heads = k_proj_weight.view(num_heads, d_k, d_model)
        v_heads = v_proj_weight.view(num_heads, d_k, d_model)
        
        head_outputs = []
        for i in range(num_heads):
            q_i = torch.matmul(in_embeddings, q_heads[i].T)  # (4, 12, 16)
            k_i = torch.matmul(in_embeddings, k_heads[i].T)
            v_i = torch.matmul(in_embeddings, v_heads[i].T)
            
            # Add batch/head dimensions for attention
            q_i = q_i.unsqueeze(1)  # (4, 1, 12, 16)
            k_i = k_i.unsqueeze(1)
            v_i = v_i.unsqueeze(1)
            
            head_out = scaled_dot_product_attention(q_i, k_i, v_i)  # (4, 1, 12, 16)
            head_outputs.append(head_out.squeeze(1))  # (4, 12, 16)
        
        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)  # (4, 12, 64)
        output = torch.matmul(concat_output, o_proj_weight.T)
        return output
    
    # Alternative: Maybe the output projection is different
    def different_output_proj():
        q = torch.matmul(in_embeddings, q_proj_weight.T)
        k = torch.matmul(in_embeddings, k_proj_weight.T)
        v = torch.matmul(in_embeddings, v_proj_weight.T)
        
        q = q.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
        k = k.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
        v = v.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
        
        attn_output = scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(batch_size, n_queries, d_model)
        
        # Maybe no output projection?
        return attn_output
        
    # Test all implementations
    results = []
    implementations = [
        ("Current", current_implementation),
        ("Split weights", split_weights_implementation),  
        ("No output proj", different_output_proj),
    ]
    
    for name, impl in implementations:
        try:
            output = impl()
            diff = torch.abs(output - expected_output)
            match_pct = (diff < 1e-6).float().mean()
            results.append((name, output, diff.max(), match_pct))
            print(f"{name:15}: max_diff={diff.max():.6f}, match={match_pct:.3%}, first={output[0,0,:3]}")
        except Exception as e:
            print(f"{name:15}: ERROR - {e}")
    
    # Check if any came close
    best = min(results, key=lambda x: x[2])
    print(f"\nBest implementation: {best[0]} with max_diff={best[2]:.6f}")
    print(f"Expected first values: {expected_output[0,0,:5]}")

if __name__ == "__main__":
    test_different_mha_interpretations()