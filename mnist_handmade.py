import tensorflow as tf

##
## NaiveDense
##

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)
        
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]

##
## NaiveSequential
##

class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)
        return x
    @property 
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights

##
## BatchGenerator
##

import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

##
## Optimizers
##

import argparse

class NaiveOptimizer:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            v.assign_sub(g * self.learning_rate)

def create_optimizer(type, learning_rate=1e-3):
    if type == 'naive':
        return NaiveOptimizer(learning_rate)
    elif type == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate)
    elif type == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate)
    else:
        raise ValueError("Invalid optimizer type. Expected 'naive', 'sgd', or 'rmsprop'.")

# Create the parser
parser = argparse.ArgumentParser(description='Choose an optimizer for the neural network.')

# Add the optimizer argument
parser.add_argument('--optimizer', choices=['naive', 'sgd', 'rmsprop'], required=True, help='The optimizer to use: naive, sgd, or rmsprop')

# Parse the arguments
args = parser.parse_args()

# Select the optimizer function based on the command line argument
selected_optimizer = create_optimizer(args.optimizer)

##
## Training
##

def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    selected_optimizer.apply_gradients(zip(gradients, model.weights))
    return average_loss

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

##
## define the model
##

model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
]) 
assert len(model.weights) == 4

##
## load data and train the modal
##

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)

##
## evaluate the model
##

import numpy as np

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
