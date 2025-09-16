#!/usr/bin/env python3
"""
Test direct adapter implementation
"""
import torch
import numpy as np
import json
from tests.adapters import run_multihead_self_attention

def test_adapter_directly():
    """Test the adapter implementation directly"""
    
    # Load expected output
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    # Load state dict (same as conftest.py)
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Get weights (move to CPU)
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # Create test input (same as test fixture)
    batch_size = 4
    n_queries = 12  
    d_model = 64
    num_heads = 4
    torch.manual_seed(4)
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Calling adapter function directly...")
    
    # Call the adapter function directly
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    
    print(f"Actual output shape: {actual_output.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Actual range: [{actual_output.min():.6f}, {actual_output.max():.6f}]")
    print(f"Expected range: [{expected_output.min():.6f}, {expected_output.max():.6f}]")
    
    # Compare
    diff = torch.abs(actual_output - expected_output)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    # Test with original pytest tolerance
    try:
        np.testing.assert_allclose(actual_output.numpy(), expected_output.numpy(), rtol=1e-4, atol=1e-6)
        print("✅ Test PASSED!")
    except AssertionError as e:
        print("❌ Test FAILED:")
        print(str(e))

if __name__ == "__main__":
    test_adapter_directly()