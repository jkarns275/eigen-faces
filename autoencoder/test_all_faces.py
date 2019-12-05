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

plt.subplots_adjust(left=0.25, bottom=0.25)

height, width = att_faces_util.att_faces_dims()
model, em, dm = create_arch((height, width, 1))

model.load_weights('autoencoder.h5', by_name=True)
em.load_weights('autoencoder.h5', by_name=True)
dm.load_weights('autoencoder.h5', by_name=True)

att_faces = att_faces_util.load_att_faces('att_faces')
encoded_faces = em.predict(att_faces)
print(encoded_faces)

reconstructed_faces = model.predict(att_faces)

n = 100 # figure with 20x20 samples

figure = np.zeros((height * 20, width * 20))
rfigure = np.zeros((height * 20, width * 20))

ax.imshow(latent_figure, cmap='gray', vmin=0, vmax=1)
plt.show()

for i in range(400):
    column = i % 20
    row = i // 20
    rf = reconstructed_faces[i, :, :, :].reshape(height, width)
    rfigure[ height * row:height * row + height, 
            width * column:width * column + width] = rf[:, :]

for i in range(400):
    column = i % 20
    row = i // 20
    rf = att_faces[i, :, :, :].reshape(height, width)
    figure[ height * row:height * row + height, 
            width * column:width * column + width] = rf[:, :]

plt.imshow(rfigure, cmap='gray', vmin=0, vmax=1)
plt.figure()
plt.imshow(figure, cmap='gray', vmin=0, vmax=1)
plt.show()
