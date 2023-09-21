import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.datasets import mnist

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to be between 0 and 1
test_images = test_images / 255.0

# Reshape the test images for the model
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Load the model
model = tf.keras.models.load_model('examples/codeepneat/codeepneat_mnist_example/best_genome_model/')

# Make predictions on the test set
pred_labels = model.predict(test_images)

# Convert prediction probabilities to class labels
pred_labels = tf.argmax(pred_labels, axis=1)

# Calculate accuracy
accuracy = np.sum(pred_labels == test_labels) / len(test_labels)

print(f"Test accuracy: {accuracy * 100}%")
