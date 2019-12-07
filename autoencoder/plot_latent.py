import sys
import warnings
import os
import glob
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
encoded_faces = em.predict(att_faces)

print(encoded_faces)

latent_reps = {}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(4):
    start = 10 * i
    end = start + 10
    l = encoded_faces[start:end]
    x, y, z = l[:, 0], l[:, 1], l[:, 2]
    ax.scatter(x, y, z)

plt.show()
