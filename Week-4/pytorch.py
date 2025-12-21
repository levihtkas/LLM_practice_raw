```python
import torch
import time

# Data - convert to torch tensors on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand(1_000_000, device=device)
w = torch.rand(1_000_000, device=device)
b = 0.1

# Timing
start = time.perf_counter()

y = torch.dot(x, w) + b

end = time.perf_counter()

print("PyTorch Output:", y.item())
print("PyTorch Time Taken:", end - start, "seconds")
```