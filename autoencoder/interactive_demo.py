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

import att_faces_util

height, width = att_faces_util.att_faces_dims()
model, em, dm = create_arch((height, width, 1))

model.load_weights('autoencoder.h5', by_name=True)
em.load_weights('autoencoder.h5', by_name=True)
dm.load_weights('autoencoder.h5', by_name=True)

att_faces = att_faces_util.load_att_faces('att_faces')

fig, ax = plt.subplots()
latent_figure = np.zeros((height, width))

n_latent_vars = att_faces_util.n_latent_vars()
latent_axes = list(map(lambda i: plt.axes([0.25, 0.0125 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow'), range(n_latent_vars)))
sliders = list(map(lambda i: Slider(latent_axes[i], f"latent {i}", -3, 3, valinit=0, valstep=0.01), range(n_latent_vars)))

def update(value):
    z_sample = np.array(list(map(lambda s: s.val, sliders)))
    z_sample = z_sample.reshape(1, n_latent_vars)
    digit = dm.predict(z_sample, batch_size=1).reshape(height, width)
    latent_figure[:, :] = digit
    ax.imshow(latent_figure, cmap='gray', vmin=0, vmax=1)

list(map(lambda s: s.on_changed(update), sliders))

ax.imshow(latent_figure, cmap='gray', vmin=0, vmax=1)
plt.show()
