import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU device name:", torch.cuda.get_device_name(0))

# Tensor ops on GPU
a = torch.rand(10000, 10000, device="cuda")
b = torch.rand(10000, 10000, device="cuda")
c = torch.matmul(a, b)

print("Matrix multiplication complete. Result shape:", c.shape)
