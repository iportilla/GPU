# TensorFlow GPU Examples (WSL2)

This repository contains deep learning examples using TensorFlow with GPU acceleration inside WSL2 on Windows.

## Examples Included
1. `01_sentiment_analysis.py` - Sentiment analysis using imdb and cnn
2. `02_cifar10_cnn.py` - Image classification using a cnn on cifar-10
3. `03_mobilenetv2_transfer.py` - Transfer learning with mobilenetv2 on cifar-10
4. `04_softmax_fashion_mnist.py` - Multi-class classification on fashion mnist
5. `05_bert_ner_pipeline.py` - Named entity recognition using bert from hugging face
6. `06_lstm_timeseries_prediction.py` - Time-series forecasting using lstm

## Requirements
```bash
pip install tensorflow transformers datasets matplotlib seqeval

pip install transformers datasets seqeval

pip install evaluate
```

## Run
```bash
python <filename>.py
```
