import torch
import time

# Select fastest device available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64  # match NumPy default precision

# Data
x = torch.rand(1_000_000, dtype=dtype, device=device)
w = torch.rand(1_000_000, dtype=dtype, device=device)
b = torch.tensor(0.1, dtype=dtype, device=device)

# Timing
if device == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()

y = torch.dot(x, w) + b

if device == "cuda":
    torch.cuda.synchronize()
end = time.perf_counter()

print("PyTorch Output:", y.item())
print("PyTorch Time Taken:", end - start, "seconds")