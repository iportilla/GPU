# multi-class classification on Fashion MNIST

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize input images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model: Flatten → Dense → ReLU → Dense → Softmax
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {acc:.2f}")
