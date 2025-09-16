#!/usr/bin/env python3
"""
Debug weight loading and usage carefully
"""
import torch
import numpy as np
import json

def debug_weight_loading():
    """Debug how weights are loaded and used"""
    
    # Load state dict exactly like the test
    from pathlib import Path
    from tests.common import FIXTURES_PATH
    
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    print("Available attention keys:")
    attn_keys = [k for k in state_dict.keys() if 'layers.0.attn' in k]
    for key in attn_keys:
        weight = state_dict[key]
        print(f"{key}: {weight.shape}")
    
    # Get the exact weights the test uses
    print("\nGetting weights exactly like the test:")
    d, _ = (state_dict, config)  # This is exactly what ts_state_dict fixture returns
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    
    print(f"Q: {q_proj_weight.shape}, device: {q_proj_weight.device}")
    print(f"K: {k_proj_weight.shape}, device: {k_proj_weight.device}")  
    print(f"V: {v_proj_weight.shape}, device: {v_proj_weight.device}")
    print(f"O: {o_proj_weight.shape}, device: {o_proj_weight.device}")
    
    # Create input exactly like test
    batch_size = 4  # from fixture
    n_queries = 12  # from fixture  
    d_model = 64   # from fixture
    torch.manual_seed(4)  # from fixture
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print(f"\nInput: {in_embeddings.shape}, device: {in_embeddings.device}")
    print(f"Input first: {in_embeddings[0, 0, :5]}")
    
    # Check device compatibility
    if q_proj_weight.device != in_embeddings.device:
        print(f"❌ Device mismatch! Weights on {q_proj_weight.device}, input on {in_embeddings.device}")
        q_proj_weight = q_proj_weight.to(in_embeddings.device)
        k_proj_weight = k_proj_weight.to(in_embeddings.device)
        v_proj_weight = v_proj_weight.to(in_embeddings.device)
        o_proj_weight = o_proj_weight.to(in_embeddings.device)
        print("✅ Moved weights to same device")
    else:
        print("✅ Weights and input on same device")
    
    # Now call the exact adapter function
    from tests.adapters import run_multihead_self_attention
    
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=4,  # from fixture
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    print(f"\nOutput comparison:")
    print(f"Actual:   {actual_output[0, 0, :5]}")
    print(f"Expected: {expected_output[0, 0, :5]}")
    
    # Let me also check if there's anything special about the weights
    print(f"\nWeight statistics:")
    print(f"Q weight mean: {q_proj_weight.mean():.6f}, std: {q_proj_weight.std():.6f}")
    print(f"K weight mean: {k_proj_weight.mean():.6f}, std: {k_proj_weight.std():.6f}")
    print(f"V weight mean: {v_proj_weight.mean():.6f}, std: {v_proj_weight.std():.6f}")
    print(f"O weight mean: {o_proj_weight.mean():.6f}, std: {o_proj_weight.std():.6f}")
    
    # Check if weights are identity or special in some way
    print(f"\nIs Q weight identity? {torch.allclose(q_proj_weight, torch.eye(64), atol=0.1)}")
    print(f"Is K weight identity? {torch.allclose(k_proj_weight, torch.eye(64), atol=0.1)}")
    print(f"Is V weight identity? {torch.allclose(v_proj_weight, torch.eye(64), atol=0.1)}")
    print(f"Is O weight identity? {torch.allclose(o_proj_weight, torch.eye(64), atol=0.1)}")

if __name__ == "__main__":
    debug_weight_loading()