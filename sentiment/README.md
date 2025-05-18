## Setup

Install Tensorflow with:

```
pip install tensorflow==2.15.0  # or latest available with GPU support
```

# Sentiment Analysis with TensorFlow + GPU on WSL

This project demonstrates sentiment classification using the **IMDB dataset** and **TensorFlow with GPU acceleration** via **WSL2 + NVIDIA GPU**.

---

## Requirements

- Windows with WSL2 (Ubuntu 20.04 or 22.04)
- NVIDIA GPU (e.g., RTX A2000 Laptop GPU)
- CUDA & cuDNN support in WSL
- TensorFlow 2.x with GPU support

---

## Setup

### 1. Create environment & install dependencies

```bash
pip install -r requirements.txt
```

## Run code:

```bash
python sentiment\sentiment_analysis.py
```

## Example output:

```
Num GPUs Available: 1
TensorFlow version: 2.15.0
...
Epoch 1/3
...
Test accuracy: 0.87
```
