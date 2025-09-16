#!/usr/bin/env python3
"""
Test with the correct input
"""
import torch
import numpy as np
import json
from tests.adapters import run_multihead_self_attention

def test_with_correct_input():
    """Test with the correct input that matches the fixture"""
    
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
    
    # Create the CORRECT test input (same as fixture)
    torch.manual_seed(4)
    batch_size = 4
    n_queries = 12  
    d_model = 64
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("Using correct input:")
    print(f"Input first values: {in_embeddings[0, 0, :5]}")
    print(f"Expected: tensor([-0.9414,  1.2632, -0.1838,  0.1505,  0.1075])")
    
    # Verify we have the right input
    expected_first = torch.tensor([-0.9414,  1.2632, -0.1838,  0.1505,  0.1075])
    if torch.allclose(in_embeddings[0, 0, :5], expected_first, atol=1e-4):
        print("✅ Input matches test fixture!")
    else:
        print("❌ Input does not match test fixture")
        return
    
    # Call the adapter function
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=4,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    
    print(f"\nActual output shape: {actual_output.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Actual first values: {actual_output[0, 0, :5]}")  
    print(f"Expected first values: {expected_output[0, 0, :5]}")
    
    # Compare
    diff = torch.abs(actual_output - expected_output)
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}") 
    print(f"Matching elements (1e-6): {(diff < 1e-6).float().mean():.3%}")
    
    # Test with pytest tolerance
    try:
        np.testing.assert_allclose(actual_output.numpy(), expected_output.numpy(), rtol=1e-4, atol=1e-6)
        print("✅ Test PASSED!")
        return True
    except AssertionError as e:
        print("❌ Test FAILED:")
        print(str(e)[:300])
        return False

if __name__ == "__main__":
    test_with_correct_input()