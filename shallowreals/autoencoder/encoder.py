import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Input, Reshape, Conv2D
from tensorflow.keras.models import Sequential

class Encoder(Layer):
    """Maps MNIST digits to a triplet"""
    def __init__(self, input_shape=(64, 64, 3), lowmem=True, name='Encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self._input_shape = input_shape
        self._lowmem = lowmem
        self._encoder_dim = 512 if lowmem else 1024

        self.encoder_conv = Sequential([
                Input(shape=self._input_shape),
                Conv2D(128, kernel_size=5, strides=2, padding="same"),
                Conv2D(256, kernel_size=5, strides=2, padding="same"),
                Conv2D(512, kernel_size=5, strides=2, padding="same")
            ])

        self.encoder_reshape = Sequential([
                Flatten(),
                Dense(self._encoder_dim),
                Dense(4 * 4 * 1024),
                Reshape((4, 4, 1024)),
                Conv2D(2048, kernel_size=3, strides=(1, 1), padding="same")
            ])

    def call(self, inputs):
        conv = self.encoder_conv(inputs)

        if not self._lowmem:
            conv = Conv2D(1024, kernel_size=5, strides=2, padding="same")(conv)

        return self.encoder_reshape(conv)

if __name__ == "__main__":
    # test
    x = tf.random.normal((3, 28, 28, 1))
    layer = Encoder(input_shape=(28, 28, 1), lowmem=False)
    y = layer(x)
    print(y)