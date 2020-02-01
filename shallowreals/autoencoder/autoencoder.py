import tensorflow as tf
from tensorflow.keras import layers, models

from encoder import Encoder
from decoder import Decoder

class AutoEncoder(models.Model):

    def __init__(self,
                input_shape=(28,28), 
                z_dim=3, output_dim=28*28, 
                name='autoencoder', 
                **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(input_shape=input_shape, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, output_dim=output_dim)

    def call(self, inputs):
        z = self.encoder(inputs)
        rec = self.decoder(z)

        return rec

if __name__ == "__main__":
    # test
    x = tf.random.normal((5, 28,28))
    model = AutoEncoder()
    y = model(x)
    print(y)    