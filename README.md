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

## Step 2: Install & configure drivers

```bash
wsl
nvidia-smi
```

## Step 3: Install CUDA toolkit in WSL

```bash
sudo apt update
sudo apt install -y build-essential

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda
```

## Step 4: Install Conda and PyTorch with CUDA

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

restart terminal

```bash
conda create -n dl-gpu python=3.10
conda activate dl-gpu
```

# Install PyTorch with CUDA support

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Step 5: Run PyTorch GPU test

```
python code/gpu_test.py
```

## Expected output:

```
PyTorch version: 2.x
CUDA available: True
GPU device name: NVIDIA RTX A2000 Laptop GPU
Matrix multiplication complete. Result shape: torch.Size([10000, 10000])
```
