import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Parameters
vocab_size = 10000
maxlen = 500

# Load IMDB data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build model
model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=maxlen),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Train
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {acc:.2f}")
