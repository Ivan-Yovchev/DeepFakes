import tensorflow as tf
from tensorflow.keras import layers, models

class Decoder(layers.Layer):
    """Maps triplet back to original dim"""
    def __init__(self, z_dim=3, output_dim=28*28, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder = models.Sequential([
                layers.Dense(12, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(128, activation='tanh'),
                layers.Dense(output_dim, activation='sigmoid')
            ])

    def call(self, inputs):
        return self.decoder(inputs)

if __name__ == "__main__":
    # test
    x = tf.random.normal((5, 3))
    layer = Decoder()
    y = layer(x)
    print(y)