import sys
import warnings
import os
import glob
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.metrics import *
from keras.optimizers import Adam, RMSprop
from scipy.stats import norm
from keras.preprocessing import image
from keras import datasets

from keras import backend as K

from imgaug import augmenters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_config)

from create_arch import create_arch

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

model, em, dm = create_arch()

model.load_weights('autoencoder.h5', by_name=True)
em.load_weights('autoencoder.h5', by_name=True)
dm.load_weights('autoencoder.h5', by_name=True)
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

## normalize and reshape
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Lets add sample noise - Salt and Pepper
noise = augmenters.SaltAndPepper(0.1)
seq_object = augmenters.Sequential([noise])

train_x_n = seq_object.augment_images(x_train * 255) / 255
val_x_n = seq_object.augment_images(x_test * 255) / 255

n = 100 # figure with 20x20 samples
digit_size = 28

figure = np.zeros((digit_size, digit_size))

n_latent_vars = 2
latent_axes = list(map(lambda i: plt.axes([0.25, 0.1 + 0.05 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow'), range(n_latent_vars)))
sliders = list(map(lambda i: Slider(latent_axes[i], f"latent {i}", -2, 2, valinit=0, valstep=0.01), range(n_latent_vars)))

print(list(map(lambda s: s.val, sliders)))
def update(value):
    z_sample = np.array(list(map(lambda s: s.val, sliders)))
    z_sample = z_sample.reshape(1, 2)
    x_decoded = dm.predict(z_sample, batch_size=1)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    figure[0:digit_size,0:digit_size] = digit
    ax.imshow(figure, cmap='gray', vmin=0, vmax=1)

list(map(lambda s: s.on_changed(update), sliders))

ax.imshow(figure, cmap='gray', vmin=0, vmax=1)
plt.show()

# figure = np.zeros((digit_size * n, digit_size * n))
# 
# # Construct grid of latent variable values - can change values here to generate different things
# grid_x = np.linspace(-2, 2, n)
# grid_y = np.linspace(-2, 2, n)
# 
# preds = model.predict(val_x_n[:])
# encoded = em.predict(val_x_n[:])
# # decode for each square in the grid
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         # noise = (np.random.rand(*z_sample.shape) - 0.5) / 5
#         z_sample = np.array([xi, yi])
#         z_sample = z_sample.reshape(1, 2)
#         
#         x_decoded = dm.predict(z_sample, batch_size=1)
#         
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit
# 
# plt.figure(figsize=(n, n))
# plt.imshow(figure, cmap='gray', vmin=0, vmax=1)
# plt.show()

