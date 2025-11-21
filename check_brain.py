import torch
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"Numpy Version: {np.__version__}")

# Test a tiny tensor operation
x = torch.rand(5, 3)
print("Tensor Test:\n", x)
print("\nSUCCESS: The Brain is ready to be built.")