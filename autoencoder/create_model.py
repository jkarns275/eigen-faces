## load the libraries 
import sys
import warnings
import os
import glob
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model, Sequential, clone_model
from keras.metrics import *
from keras.optimizers import Adam, RMSprop
from scipy.stats import norm
from keras.preprocessing import image
from keras import datasets

from keras import backend as K

from imgaug import augmenters
import matplotlib.pyplot as plt

from create_arch import create_arch
import att_faces_util

session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_config)

height, width = att_faces_util.att_faces_dims()

att_faces = att_faces_util.load_att_faces('att_faces').reshape(400, height, width)

# test_subjects = list(range(10)) # The 10 images from each of these test subjects will be test data
# 
# x_test = np.zeros((10 * len(test_subjects), height, width))
# 
# for i, subject in enumerate(test_subjects):
#     for j in range(10):
#         index = subject * 10 + j
#         test_index = i * 10 + j
#         x_test[test_index, :, :] = att_faces[index, :, :]
# 
# train_subjects = list(set(range(40)) - set(test_subjects))
# 
# x_train = np.zeros((10 * len(train_subjects), height, width))
# 
# for i, subject in enumerate(train_subjects):
#     for j in range(10):
#         index = subject * 10 + j
#         train_index = i * 10 + j
#         x_train[train_index, :, :] = att_faces[index, :, :]

# 2/10 of the images are test
x_test = np.zeros((2 * 40, height, width))
x_train = np.zeros((8 * 40, height, width))

for i in range(40):
    si = 10 * i
    x_test[i * 2:2 + i * 2, :, :] = att_faces[si:si+2, :, :]
    x_train[i * 8:8 + i * 8, :, :] = att_faces[si:si+8, :, :]
    

model, encoder, decoder = create_arch((height, width, 1))

model.summary()

batch_size = 10

print(x_train.shape)

x_train = x_train.reshape(8 * 40, height, width, 1)
x_test = x_test.reshape(2 * 40, height, width, 1)

# Train autoencoder
model.fit(x=x_train, y=None, shuffle=True,
        epochs=80, 
        batch_size=batch_size, 
        validation_data=(x_test, None))

model.save_weights('autoencoder.h5')
