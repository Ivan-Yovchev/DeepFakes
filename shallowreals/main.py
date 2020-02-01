import sys
import tensorflow as tf

sys.path.insert(0, './autoencoder')
from autoencoder import AutoEncoder

if __name__ == "__main__":
    x = tf.random.normal((5, 28,28))
    model = AutoEncoder()
    y = model(x)
    print(y)