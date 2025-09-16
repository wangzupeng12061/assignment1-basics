#!/usr/bin/env python3
"""
Final attempt to debug multi-head attention by checking if there's a bias issue
or initialization problem we missed
"""
import torch
import numpy as np
import json
from cs336_basics.model import MultiHeadSelfAttention, scaled_dot_product_attention

def debug_mha_final():
    """最后的调试尝试：检查是否有bias或初始化问题"""
    
    # Load expected output and fixtures
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    from pathlib import Path
    from tests.common import FIXTURES_PATH
    
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    print("=== 检查权重是否有bias ===")
    
    # 检查是否有bias权重
    bias_keys = [k for k in state_dict.keys() if 'bias' in k and 'layers.0.attn' in k]
    print(f"Found bias keys: {bias_keys}")
    
    # 获取权重
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # 检查是否有对应的bias
    q_bias = state_dict.get('layers.0.attn.q_proj.bias', None)
    k_bias = state_dict.get('layers.0.attn.k_proj.bias', None)
    v_bias = state_dict.get('layers.0.attn.v_proj.bias', None)
    o_bias = state_dict.get('layers.0.attn.output_proj.bias', None)
    
    print(f"Q bias: {q_bias is not None}")
    print(f"K bias: {k_bias is not None}")
    print(f"V bias: {v_bias is not None}")
    print(f"O bias: {o_bias is not None}")
    
    # 创建输入
    torch.manual_seed(4)
    batch_size, n_queries, d_model = 4, 12, 64
    num_heads = 4
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("\n=== 测试带bias的实现 ===")
    
    # 使用MultiHeadSelfAttention类，设置正确的bias
    mha_with_bias = MultiHeadSelfAttention(d_model, num_heads, bias=True)
    mha_with_bias.q_proj.weight.data = q_proj_weight
    mha_with_bias.k_proj.weight.data = k_proj_weight
    mha_with_bias.v_proj.weight.data = v_proj_weight
    mha_with_bias.output_proj.weight.data = o_proj_weight
    
    # 设置bias（如果存在）
    if q_bias is not None:
        mha_with_bias.q_proj.bias.data = q_bias
    if k_bias is not None:
        mha_with_bias.k_proj.bias.data = k_bias
    if v_bias is not None:
        mha_with_bias.v_proj.bias.data = v_bias
    if o_bias is not None:
        mha_with_bias.output_proj.bias.data = o_bias
    
    output_with_bias = mha_with_bias(in_embeddings)
    
    # 测试不带bias的实现
    mha_no_bias = MultiHeadSelfAttention(d_model, num_heads, bias=False)
    mha_no_bias.q_proj.weight.data = q_proj_weight
    mha_no_bias.k_proj.weight.data = k_proj_weight
    mha_no_bias.v_proj.weight.data = v_proj_weight
    mha_no_bias.output_proj.weight.data = o_proj_weight
    
    output_no_bias = mha_no_bias(in_embeddings)
    
    print(f"Expected first: {expected_output[0, 0, :5]}")
    print(f"With bias first: {output_with_bias[0, 0, :5]}")
    print(f"No bias first: {output_no_bias[0, 0, :5]}")
    
    diff_bias = torch.abs(output_with_bias - expected_output)
    diff_no_bias = torch.abs(output_no_bias - expected_output)
    
    print(f"Bias version max diff: {diff_bias.max():.6f}, match: {(diff_bias < 1e-6).float().mean():.3%}")
    print(f"No bias version max diff: {diff_no_bias.max():.6f}, match: {(diff_no_bias < 1e-6).float().mean():.3%}")
    
    # 尝试不同的随机种子
    print("\n=== 测试不同随机种子的影响 ===")
    
    for seed in [1, 2, 3, 4, 5]:
        torch.manual_seed(seed)
        test_input = torch.randn(batch_size, n_queries, d_model)
        test_output = mha_no_bias(test_input)
        
        # 检查输出范围
        print(f"Seed {seed}: output range [{test_output.min():.3f}, {test_output.max():.3f}], first: {test_output[0,0,:3]}")
    
    print(f"Expected range: [{expected_output.min():.3f}, {expected_output.max():.3f}]")
    
    # 检查权重的数值特征
    print("\n=== 权重数值特征 ===")
    print(f"Q weight: mean={q_proj_weight.mean():.6f}, std={q_proj_weight.std():.6f}")
    print(f"K weight: mean={k_proj_weight.mean():.6f}, std={k_proj_weight.std():.6f}")
    print(f"V weight: mean={v_proj_weight.mean():.6f}, std={v_proj_weight.std():.6f}")
    print(f"O weight: mean={o_proj_weight.mean():.6f}, std={o_proj_weight.std():.6f}")
    
    return output_no_bias, expected_output

if __name__ == "__main__":
    debug_mha_final()