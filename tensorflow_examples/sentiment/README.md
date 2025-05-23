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


## Troubleshooting

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get install cuda-toolkit-12-3 libcudnn8 libcudnn8-dev -y

echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version         # Should show CUDA 12.3
nvidia-smi             # Should show your A2000 GPU


ls /usr/local/cuda-12.3/lib64/libcudnn*
ls /usr/local/cuda-12.3/lib64/libcublas*
ls /usr/local/cuda-12.3/lib64/libcufft*

pip uninstall tensorflow -y
pip install tensorflow==2.15.0

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

sudo apt update

sudo apt install libcudnn8 libcudnn8-dev -y

ls /usr/lib/x86_64-linux-gnu/libcudnn*

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

TF_CPP_MIN_LOG_LEVEL=0 python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


# Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update


sudo apt install -y cuda-toolkit-12-3 libcudnn8 libcudnn8-dev

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/.bashrc

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

TF_CPP_MIN_LOG_LEVEL=0 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

## run test

python

```bash
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
```

### Expected result

```
TensorFlow version: 2.15.0
Available GPUs: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
```
