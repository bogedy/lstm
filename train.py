import torch
import os
from pathlib import Path
import time

# Environment detection
is_colab = os.environ.get('COLAB_GPU') is not None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data directory setup (placeholder)
data_dir = Path('/content/data') if is_colab else Path('data')
data_dir.mkdir(exist_ok=True)  # Will create dir if needed
print(f"Data directory: {data_dir}")  # Verify path (you'll use this later)

start = time.time()
# Simple GPU/CPU test
x = torch.randn(1000, 1000).to(device)
for _ in range(200):
    x @ x  # Matrix multiplication benchmark

elapsed = time.time() - start

print(f"Test completed on {device} in {elapsed:.3f} seconds.")