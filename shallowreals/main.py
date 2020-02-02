import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid

# import autoencoder dependencies
sys.path.insert(0, './autoencoder')
from autoencoder import AutoEncoder

if __name__ == "__main__":
    # load MNIST dataset
    mnist = tf.keras.datasets.mnist

    # extract train and val data
    (x_train, _),(x_val, _) = mnist.load_data()

    # reshape and normalize in range [0 .. 1]
    x_train, x_val = x_train.reshape(-1, 28*28) / 255.0, x_val.reshape(-1, 28*28) / 255.0

    # init model
    model = AutoEncoder(z_dim=32)

    # set loss and optimizer type
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train model
    model.fit(x_train, x_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_val, x_val))

    # show some results
    # =================== PLOTTING ============================

    # images per row and col
    NUM_IMG_PER_ROW = 10

    # to store images
    selected_imgs = []

    # pick random indexes to visualize
    indexes = np.random.random_integers(x_train.shape[0], size=(1,NUM_IMG_PER_ROW**2))

    # add to list
    for i in range(len(indexes)):
        selected_imgs.append(x_train[indexes[i]])
    selected_imgs = tf.stack(selected_imgs)[0]

    # forwads pass
    selected_imgs = model(selected_imgs)

    # create grid
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