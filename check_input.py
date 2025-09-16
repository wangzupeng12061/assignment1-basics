#!/usr/bin/env python3
"""
Check the exact input generation from the test
"""
import torch
import json

def check_input_generation():
    """Check how the test generates input"""
    
    # From the test output, we see the input should start with [-0.9414,  1.2632, ...]
    # But my generation with seed 4 gives different values
    
    print("Testing different fixture combinations:")
    
    # Test 1: Basic seed 4 with different batch sizes
    for batch_size in [1, 4]:
        for n_queries in [12, 16]:
            for d_model in [64]:
                torch.manual_seed(4)
                in_embeddings = torch.randn(batch_size, n_queries, d_model)
                print(f"batch={batch_size}, queries={n_queries}, d_model={d_model}: {in_embeddings[0, 0, :5]}")
    
    # Maybe the conftest.py uses a different random order?
    print("\nTesting the exact fixture creation order:")
    
    # Recreate fixtures in the exact order they appear in conftest.py
    # Let me check what order the fixtures are called in
    
    # First batch_size fixture
    batch_size = 4  # from @pytest.fixture def batch_size(): return 4
    
    # Then other fixtures... Let me check if there are other random calls before in_embeddings
    
    # Check if there are other fixtures that use random numbers before in_embeddings
    print("\nTrying with potential other random calls first:")
    
    torch.manual_seed(4)
    # Maybe there are other fixtures using randomness before in_embeddings?
    # Let's see... in conftest.py there could be other fixtures
    
    # Let me simulate potential other random calls
    for i in range(10):  # Try up to 10 other potential random calls
        torch.manual_seed(4)
        for j in range(i):
            dummy = torch.randn(1)  # Simulate other random calls
        
        batch_size = 4
        n_queries = 12  
        d_model = 64
        in_embeddings = torch.randn(batch_size, n_queries, d_model)
        
        if torch.allclose(in_embeddings[0, 0, :5], torch.tensor([-0.9414,  1.2632, -0.1838,  0.1505,  0.1075]), atol=1e-4):
            print(f"Found match with {j} prior random calls!")
            print(f"Input: {in_embeddings[0, 0, :5]}")
            return in_embeddings
        else:
            print(f"With {j} prior calls: {in_embeddings[0, 0, :5]}")
    
    print("No match found within 10 iterations")
    return None

if __name__ == "__main__":
    correct_input = check_input_generation()