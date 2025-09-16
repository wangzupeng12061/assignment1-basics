#!/usr/bin/env python3
"""
Compare Linear layer behavior with multi-head attention weight usage
"""
import torch
import numpy as np
import json
from tests.adapters import run_linear, run_multihead_self_attention
from cs336_basics.model import Linear

def compare_linear_behavior():
    """Compare how Linear layer works with how I use weights in attention"""
    
    # Load state dict
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get a linear weight to test
    w1_weight = state_dict['layers.0.ffn.w1.weight'].cpu()  # [128, 64]
    
    # Create test input
    torch.manual_seed(4)
    d_model = 64
    d_ff = 128
    batch_size = 4
    n_queries = 12
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Testing Linear layer behavior:")
    print(f"W1 weight shape: {w1_weight.shape}")
    print(f"Input shape: {in_embeddings.shape}")
    
    # Method 1: Using adapter (which uses Linear class)
    output1 = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    
    # Method 2: Direct matrix multiplication (like I do in attention)
    output2 = torch.matmul(in_embeddings, w1_weight.T)
    
    # Method 3: Manual Linear class usage
    linear = Linear(d_model, d_ff, bias=False)
    linear.weight.data = w1_weight
    output3 = linear(in_embeddings)
    
    print(f"Adapter output shape: {output1.shape}")
    print(f"Direct matmul shape: {output2.shape}")
    print(f"Manual Linear shape: {output3.shape}")
    
    print(f"Adapter vs Direct diff: {torch.abs(output1 - output2).max():.6f}")
    print(f"Adapter vs Manual diff: {torch.abs(output1 - output3).max():.6f}")
    print(f"Direct vs Manual diff: {torch.abs(output2 - output3).max():.6f}")
    
    print("\nFirst few values comparison:")
    print(f"Adapter: {output1[0, 0, :5]}")
    print(f"Direct:  {output2[0, 0, :5]}")
    print(f"Manual:  {output3[0, 0, :5]}")
    
    # Now let's check if this same pattern works for attention weights
    print("\n" + "="*50)
    print("Testing attention projection behavior:")
    
    # Get attention weights
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()  # [64, 64]
    print(f"Q weight shape: {q_proj_weight.shape}")
    
    # Method 1: Direct matmul (what I currently do)
    q_direct = torch.matmul(in_embeddings, q_proj_weight.T)
    
    # Method 2: Using Linear class (like the Linear adapter does)
    q_linear = Linear(d_model, d_model, bias=False)
    q_linear.weight.data = q_proj_weight
    q_linear_output = q_linear(in_embeddings)
    
    print(f"Q Direct shape: {q_direct.shape}")
    print(f"Q Linear shape: {q_linear_output.shape}")
    print(f"Q Direct vs Linear diff: {torch.abs(q_direct - q_linear_output).max():.6f}")
    print(f"Q Direct first: {q_direct[0, 0, :5]}")
    print(f"Q Linear first: {q_linear_output[0, 0, :5]}")
    
    # They should be the same since both use y = x @ W^T
    if torch.allclose(q_direct, q_linear_output, atol=1e-6):
        print("✅ Direct matmul and Linear class give same results")
    else:
        print("❌ Direct matmul and Linear class give different results")

if __name__ == "__main__":
    compare_linear_behavior()