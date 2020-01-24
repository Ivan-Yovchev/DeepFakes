import tensorflow as tf
from tensorflow.keras import layers, models

class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet"""
    def __init__(self, input_shape=(28, 28), z_dim=3, name='Encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder = models.Sequential([
                layers.Flatten(input_shape=input_shape),
                layers.Dense(128, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(12, activation='tanh'),
                layers.Dense(z_dim)
            ])

    def call(self, inputs):
        return self.encoder(inputs)

# test
# x = tf.random.normal((5, 28,28))
# layer = Encoder()
# y = layer(x)
# print(y)