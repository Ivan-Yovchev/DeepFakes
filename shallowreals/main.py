import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid

# import autoencoder dependencies
sys.path.insert(0, './autoencoder')
from autoencoder import AutoEncoder

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, _),(x_val, _) = mnist.load_data()
    x_train, x_val = x_train.reshape(-1, 28*28) / 255.0, x_val.reshape(-1, 28*28) / 255.0

    model = AutoEncoder()

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, x_train, batch_size=1024, epochs=10, verbose=1, validation_data=(x_val, x_val))

    # show some results
    NUM_IMG_PER_ROW = 10
    selected_imgs = []
    indexes = np.random.random_integers(x_train.shape[0], size=(1,NUM_IMG_PER_ROW**2))
    for i in range(len(indexes)):
        selected_imgs.append(x_train[indexes[i]])
    selected_imgs = tf.stack(selected_imgs)[0]

    selected_imgs = model(selected_imgs)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(NUM_IMG_PER_ROW, NUM_IMG_PER_ROW),
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    for ax, im in zip(grid, selected_imgs):
        # Iterating over the grid returns the Axes.
        im = tf.reshape(im, [28, 28]).numpy()
        ax.imshow(im, cmap='gray')

    plt.show()