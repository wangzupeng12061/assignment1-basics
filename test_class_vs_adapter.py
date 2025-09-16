#!/usr/bin/env python3
"""
Test if using the actual MultiHeadSelfAttention class matches the adapter
"""
import torch
import numpy as np
import json
from cs336_basics.model import MultiHeadSelfAttention
from tests.adapters import run_multihead_self_attention

def test_class_vs_adapter():
    """Test if the class implementation matches the adapter"""
    
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
    
    print("Comparing MultiHeadSelfAttention class with adapter:")
    
    # Method 1: Using adapter
    adapter_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    
    # Method 2: Using the class
    mha = MultiHeadSelfAttention(d_model, num_heads, bias=False)
    mha.q_proj.weight.data = q_proj_weight
    mha.k_proj.weight.data = k_proj_weight
    mha.v_proj.weight.data = v_proj_weight
    mha.output_proj.weight.data = o_proj_weight
    
    class_output = mha(in_embeddings)
    
    print(f"Adapter output shape: {adapter_output.shape}")
    print(f"Class output shape: {class_output.shape}")
    
    print(f"Adapter first: {adapter_output[0, 0, :5]}")
    print(f"Class first: {class_output[0, 0, :5]}")
    
    diff = torch.abs(adapter_output - class_output)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    if torch.allclose(adapter_output, class_output, atol=1e-6):
        print("✅ Class and adapter implementations are identical")
    else:
        print("❌ Class and adapter implementations differ")
    
    # Now check against expected output
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    print(f"\nComparing with expected output:")
    print(f"Expected first: {expected_output[0, 0, :5]}")
    
    adapter_diff = torch.abs(adapter_output - expected_output)
    class_diff = torch.abs(class_output - expected_output)
    
    print(f"Adapter vs Expected max diff: {adapter_diff.max():.6f}")
    print(f"Class vs Expected max diff: {class_diff.max():.6f}")

if __name__ == "__main__":
    test_class_vs_adapter()