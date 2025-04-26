########TASK 2 - Deep Learning Model using TensorFlow (Image Classification)

import tensorflow as tf  # TensorFlow for building deep learning models

from tensorflow.keras import layers, models  # Layers and model building API
from tensorflow.keras.datasets import mnist  # Preloaded MNIST dataset
 
 # Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

 # Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 1D
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLUactivation
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes with softmax
])

 # Compile model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
 
 # Train model for 5 epochs with validation
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

 # Evaluate model performance on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")