import tensorflow as tf
from tensorflow import keras

layers = keras.layers


def cnn_net(inputs, kernel_init=tf.initializers.he_normal(), activation_fn=tf.nn.leaky_relu, padding="same"):
    x = layers.Conv2D(
        2, (3, 3), (2, 2), activation=activation_fn, kernel_initializer=kernel_init, padding=padding, name='conv1'
    )(inputs)
    print(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    print(x)
    x = layers.Conv2D(
        4, (3, 3), (2, 2), activation=activation_fn, kernel_initializer=kernel_init, padding=padding, name='conv2'
    )(x)
    print(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)
    print(x)
    x = layers.Conv2D(
        8, (3, 3), (2, 2), activation=activation_fn, kernel_initializer=kernel_init, padding=padding, name='conv3'
    )(x)
    print(x)
    x = layers.BatchNormalization(name="bn_conv3")(x)
    print(x)
    x = layers.Flatten()(x)
    print(x)
    x = layers.Dense(
        2, activation=tf.nn.sigmoid)(x)
    print(x)
    return x
