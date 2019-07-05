from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def train():
    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("Train_images shape:", train_images.shape)
    print("Train_labels shape:", train_labels.shape)
    print("Test_images shape:", test_images.shape)
    print("Test_labels shape:", test_labels.shape)

    #  scale these values to a range of 0 to 1 before feeding to the neural network model
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Build the model
    # Setup layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model
    # 损失函数、优化器和训练测试时的评估指标
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5)
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy: ', test_acc)

def aiTest(x_test, input_shape):

