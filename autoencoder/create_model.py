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


session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_config)


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

model, encoder, decoder = create_arch()

model.summary()

batch_size = 2048

# Train autoencoder
model.fit(x=x_train, y=None, shuffle=True, epochs=20, batch_size=batch_size, validation_data=(x_test, None))

model.save_weights('autoencoder.h5')
