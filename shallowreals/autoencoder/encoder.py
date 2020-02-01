import tensorflow as tf
from tensorflow.keras import layers, models

class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet"""
    def __init__(self, z_dim=3, name='Encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder = models.Sequential([
                layers.Dense(128, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(12, activation='tanh'),
                layers.Dense(z_dim)
            ])

    def call(self, inputs):
        return self.encoder(inputs)

if __name__ == "__main__":
    # test
    x = tf.random.normal((5, 28*28))
    layer = Encoder()
    y = layer(x)
    print(y)