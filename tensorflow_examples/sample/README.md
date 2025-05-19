# CNN on MNIST with PyTorch + GPU (WSL)

This example demonstrates training a simple Convolutional Neural Network (CNN) on the **MNIST dataset** using PyTorch with **GPU acceleration via CUDA** under **WSL2** on Windows.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
## Run code

```bash
python sample\train_mnist.py
```

## Example output

```
Using device: cuda
Epoch 1, Loss: 0.2123
Epoch 2, Loss: 0.0621
Epoch 3, Loss: 0.0457
Test accuracy: 98.76%
```
