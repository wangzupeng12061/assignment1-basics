#!/usr/bin/env python3
"""
Analyze weight structure and expected behavior
"""
import torch
import numpy as np
import json

def analyze_weights():
    """Analyze the weight structure"""
    
    # Load state dict 
    state_dict = torch.load('tests/fixtures/ts_tests/model.pt', weights_only=True)
    config = json.load(open('tests/fixtures/ts_tests/model_config.json'))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    print("Model config:", config)
    
    # Get weights
    q_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    print(f"\nWeight shapes:")
    print(f"Q: {q_weight.shape}")
    print(f"K: {k_weight.shape}")
    print(f"V: {v_weight.shape}")
    print(f"O: {o_weight.shape}")
    
    d_model = config['d_model']
    num_heads = config.get('num_heads', 4)  # Default assumption
    d_k = d_model // num_heads
    
    print(f"\nExpected dimensions:")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k: {d_k}")
    
    # Test if weights are organized by heads
    # If this is a standard multihead implementation, the weight matrix
    # should be [d_model, d_model] and represent all heads concatenated
    
    # Let's try to understand by looking at the first few values
    print(f"\nFirst few Q weight values:")
    print(q_weight[:5, :5])
    
    # Check if weights can be reshaped for multi-head
    if q_weight.shape[0] == d_model and q_weight.shape[1] == d_model:
        print(f"\nWeights are [d_model, d_model] = [{d_model}, {d_model}]")
        print("This suggests all heads are concatenated in the output dimension")
        
        # Try reshaping to see head structure
        q_heads = q_weight.view(num_heads, d_k, d_model)
        print(f"Reshaped Q to [num_heads, d_k, d_model]: {q_heads.shape}")
        
        print(f"Head 0 first row: {q_heads[0, 0, :5]}")
        print(f"Head 1 first row: {q_heads[1, 0, :5]}")

if __name__ == "__main__":
    analyze_weights()