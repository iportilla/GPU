# Deep Learning on WSL with NVIDIA GPU (A2000) + PyTorch

This repo shows how to configure a Windows laptop with an NVIDIA GPU (e.g., A2000) using **WSL2**, and run **PyTorch** deep learning workloads on the GPU.

---

## Requirements

- Windows 10/11 with WSL2 enabled
- NVIDIA GPU (e.g., RTX A2000 Laptop GPU)
- Latest NVIDIA GPU drivers with WSL2 support
- Ubuntu (WSL2, e.g., Ubuntu 22.04)

---

## Step 1: Enable WSL and Install Ubuntu

In PowerShell (Admin):

```bash
wsl --install -d Ubuntu
```

