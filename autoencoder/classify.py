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

from vae_layer import VAELayer
import att_faces_util
from create_arch import create_arch

def create_carch():
    
    latent_dim = att_faces_util.n_latent_vars()
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(2 * latent_dim, name='L1')(latent_inputs)
    x = Dense(4 * latent_dim, name='L2')(x)
    x = Dense(40, name='output')(x)
    output = Softmax(axis=-1)(x)
    
    model = Model(latent_inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def create_cmodel():
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_config)

    classifier = create_carch()
    height, width = att_faces_util.att_faces_dims()
    model, em, dm = create_arch((height, width, 1))

    model.load_weights('autoencoder.h5', by_name=True)
    em.load_weights('autoencoder.h5', by_name=True)
    dm.load_weights('autoencoder.h5', by_name=True)
    latent_dim = att_faces_util.n_latent_vars()
    att_faces = att_faces_util.load_att_faces('att_faces')
    encoded_faces = em.predict(att_faces)
    labels = np.zeros((400, 40))
    def sparse(k, size):
        a = np.zeros(size)
        a[k] = 1
        return a
    for i in range(40):
        labels[i*10:i*10 + 10, :] += sparse(i, 40)

    n_test = 2
    n_train = 10 - n_test

    x_test = np.zeros((40 * n_test, latent_dim))
    labels_test = np.zeros((40 * n_test, 40))

    x_train = np.zeros((40 * n_train, latent_dim))
    labels_train = np.zeros((40 * n_train, 40))
    
    for i in range(40):
        test_start = i * n_test
        test_end = test_start + n_test
        
        train_start = i * n_train
        train_end = train_start + n_train

        x_test[test_start:test_end, :] = encoded_faces[10 * i:10 * i + n_test, :]
        labels_test[test_start:test_end, :] = labels[10 * i:10 * i + n_test, :]

        x_train[train_start:train_end, :] = encoded_faces[10 * i + n_test:10 * i + 10, :]
        labels_train[train_start:train_end, :] = labels[10 * i + n_test:10 * i + 10, :]

    classifier.fit(x=x_train, y=labels_train, shuffle=True,
                    epochs=250,
                    batch_size=16,
                    validation_data=(x_test, labels_test))
    predictions = classifier.predict(encoded_faces)
    hits = 0
    for i in range(400):
        pred = predictions[i, :]
        identity = np.argmax(pred)
        correct = identity == (i // 10)
        hits += int(correct)
        print(f"image {i:4} -> {correct}")
    print(f"accuracy: {hits / 400}")
create_cmodel()
