# time-series forecasting using LSTM

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Generate synthetic sine wave data
def generate_data(seq_length=50, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    y = np.sin(x)
    X, Y = [], []
    for i in range(len(y) - seq_length):
        X.append(y[i:i + seq_length])
        Y.append(y[i + seq_length])
    return np.array(X), np.array(Y)

SEQ_LENGTH = 50
X, Y = generate_data(seq_length=SEQ_LENGTH)
X = np.expand_dims(X, axis=2)  # Shape: (samples, seq_len, 1)

# Train-test split
split = int(0.8 * len(X))
x_train, y_train = X[:split], Y[:split]
x_test, y_test = X[split:], Y[split:]

# LSTM model
model = models.Sequential([
    layers.Input(shape=(SEQ_LENGTH, 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predict
y_pred = model.predict(x_test)

# Plot prediction vs actual
plt.figure(figsize=(10, 4))
plt.plot(y_test, label="True")
plt.plot(y_pred.flatten(), label="Predicted")
plt.legend()
plt.title("LSTM Time-Series Forecasting")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.show()
