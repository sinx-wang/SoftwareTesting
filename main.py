from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cnn_model


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


def aiTest1(x_test, input_shape):
    batch_size = 64
    pic_num, width, height = input_shape[0], input_shape[1], input_shape[2]
    mnist_dim = width * height
    random_dim = 10
    epochs = 10

    def random_init(size):
        return tf.random_uniform(size, -0.05, 0.05)


def aiTest(x_test: list, input_shape: tuple):
    new_images = []
    model = cnn_model.load_model()
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output
    batch = 0

    for image in x_test:
        image = np.expand_dims(image, 0)
        hacked_image = np.copy(image)

        error_tag = np.argmax(model.predict(image)[0])
        error_tag -= 1
        if error_tag == -1:
            error_tag = 9

        cost_func = model_output_layer[0, error_tag]
        gradient_func = keras.backend.gradients(cost_func, model_input_layer)[0]
        grab_cost_and_gradients_from_model = keras.backend.function([model_input_layer, keras.backend.learning_phase()],
                                                                    [cost_func, gradient_func])

        e = 0.007
        cost = 0
        while cost < 0.6:
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
            n = np.sign(gradients)
            hacked_image += n * e
            hacked_image = np.clip(hacked_image, -1, 1)

        new_images.append(hacked_image[0])
        print("batch:{} cost:{:.5}%".format(batch, cost * 100))
        batch += 1
    return new_images
