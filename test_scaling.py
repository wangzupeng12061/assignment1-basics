#!/usr/bin/env python3
"""
Check if the non-RoPE test actually expects some form of position encoding or different scaling
"""
import torch
import numpy as np
import json
import math
from cs336_basics.model import MultiHeadSelfAttention, scaled_dot_product_attention

def test_scaling_hypothesis():
    """测试是否存在缩放因子问题"""
    
    # Load data
    expected_output = torch.from_numpy(np.load('tests/_snapshots/test_multihead_self_attention.npz')['array'])
    
    from pathlib import Path
    from tests.common import FIXTURES_PATH
    
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    q_proj_weight = state_dict['layers.0.attn.q_proj.weight'].cpu()
    k_proj_weight = state_dict['layers.0.attn.k_proj.weight'].cpu() 
    v_proj_weight = state_dict['layers.0.attn.v_proj.weight'].cpu()
    o_proj_weight = state_dict['layers.0.attn.output_proj.weight'].cpu()
    
    # 创建输入
    torch.manual_seed(4)
    batch_size, n_queries, d_model = 4, 12, 64
    num_heads = 4
    d_k = d_model // num_heads
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    print("=== 测试不同缩放因子 ===")
    
    # 获取我们当前的输出
    mha = MultiHeadSelfAttention(d_model, num_heads, bias=False)
    mha.q_proj.weight.data = q_proj_weight
    mha.k_proj.weight.data = k_proj_weight
    mha.v_proj.weight.data = v_proj_weight
    mha.output_proj.weight.data = o_proj_weight
    
    our_output = mha(in_embeddings)
    
    # 测试各种缩放因子
    import math
    scaling_factors = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, math.sqrt(2), 1/math.sqrt(2)]
    
    best_match = float('inf')
    best_scale = None
    
    for scale in scaling_factors:
        scaled_output = our_output * scale
        diff = torch.abs(scaled_output - expected_output).max()
        match_pct = (torch.abs(scaled_output - expected_output) < 1e-6).float().mean()
        
        print(f"Scale {scale:.3f}: max_diff={diff:.6f}, match={match_pct:.3%}")
        
        if diff < best_match:
            best_match = diff
            best_scale = scale
    
    print(f"\\nBest scale: {best_scale} with max_diff: {best_match:.6f}")
    
    # 测试权重缩放
    print("\\n=== 测试权重缩放 ===")
    
    weight_scales = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    
    for scale in weight_scales:
        mha_scaled = MultiHeadSelfAttention(d_model, num_heads, bias=False)
        mha_scaled.q_proj.weight.data = q_proj_weight * scale
        mha_scaled.k_proj.weight.data = k_proj_weight * scale  
        mha_scaled.v_proj.weight.data = v_proj_weight * scale
        mha_scaled.output_proj.weight.data = o_proj_weight * scale
        
        scaled_output = mha_scaled(in_embeddings)
        diff = torch.abs(scaled_output - expected_output).max()
        match_pct = (torch.abs(scaled_output - expected_output) < 1e-6).float().mean()
        
        print(f"Weight scale {scale:.3f}: max_diff={diff:.6f}, match={match_pct:.3%}")
    
    # 测试attention中的scaling
    print("\\n=== 测试自定义attention缩放 ===")
    
    import math
    from cs336_basics.model import scaled_dot_product_attention
    
    def custom_scaled_attention(Q, K, V, scale_factor=1.0):
        """带自定义缩放的attention"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * scale_factor)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output
    
    # 手动实现带自定义缩放
    q = torch.matmul(in_embeddings, q_proj_weight.T)
    k = torch.matmul(in_embeddings, k_proj_weight.T)
    v = torch.matmul(in_embeddings, v_proj_weight.T)
    
    q = q.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
    k = k.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
    v = v.view(batch_size, n_queries, num_heads, d_k).transpose(-3, -2)
    
    attention_scales = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    
    for scale in attention_scales:
        attn_output = custom_scaled_attention(q, k, v, scale)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(batch_size, n_queries, d_model)
        output = torch.matmul(attn_output, o_proj_weight.T)
        
        diff = torch.abs(output - expected_output).max()
        match_pct = (torch.abs(output - expected_output) < 1e-6).float().mean()
        
        print(f"Attention scale {scale:.3f}: max_diff={diff:.6f}, match={match_pct:.3%}")

if __name__ == "__main__":
    test_scaling_hypothesis()