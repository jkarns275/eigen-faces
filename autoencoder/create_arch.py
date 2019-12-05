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

def create_arch(input_shape):
    model = Sequential(name='autoencoder')
    
    batch_size = 16
    latent_dim = att_faces_util.n_latent_vars()  # Number of latent dimension parameter
    
    # Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
    input_img = Input(shape=input_shape)
    
    x = Conv2D(32, 3, name='c1',
                      padding='same', 
                      activation='relu')(input_img)
    x = Conv2D(64, 3, name = 'c2',
                      padding='same', 
                      activation='relu',
                      strides=(2, 2))(x)
    x = Conv2D(64, 3, name='c3',
                      padding='same', 
                      activation='relu')(x)
    x = Conv2D(64, 3, name='c4',
                      padding='same', 
                      activation='relu')(x)
    
    # need to know the shape of the network here for the decoder
    shape_before_flattening = K.int_shape(x)
    
    x = Flatten(name='f1')(x)
    x = Dense(32, name='d1', activation='relu')(x)
    
    # Two outputs, latent mean and (log)variance
    z_mu = Dense(latent_dim, name='d2')(x)
    z_log_sigma = Dense(latent_dim, name='d3')(x)
    
    # sampling function
    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                                  mean=0., stddev=1.)
        return z_mu + K.exp(z_log_sigma) * epsilon
    
    # sample vector from the latent distribution
    z = Lambda(sampling, name='L1')([z_mu, z_log_sigma])
    
    encoder = Model(input_img, z)

    # decoder takes the latent distribution sample as input
    decoder_input = Input(K.int_shape(z)[1:], name='decoder_input')
    
    # Expand to original size
    x = Dense(np.prod(shape_before_flattening[1:]), name='d4',
                     activation='relu')(decoder_input)
    
    # reshape
    x = Reshape(shape_before_flattening[1:], name='r1')(x)
    
    # use Conv2DTranspose to reverse the conv layers from the encoder
    x = Conv2DTranspose(32, 3, name='u1',
                        padding='same', 
                        activation='relu',
                        strides=(2, 2))(x)
    x = Conv2D(1, 3, name='c5',
                padding='same', 
                activation='sigmoid')(x)
    
    # decoder model statement
    decoder = Model(decoder_input, x)
    
    # apply the decoder to the sample from the latent distribution
    z_decoded = decoder(z)
    
    vae = VAELayer()([input_img, z_decoded, z_mu, z_log_sigma])
    model = Model(input_img, vae)
    model.compile(optimizer='rmsprop', loss=None)
    model.summary()
    
    encoder.summary()
    decoder.summary()

    return model, encoder, decoder
#     for layer in layers:
#         model.add(layer)
#     
#     # compile the model
#     model.compile(optimizer='adam', loss='mse')
#     
#     # Isolate encoder
#     encoder_input = InputLayer(input_shape=(28, 28, 1), name='encoder_input')
#     encoder_layers = [encoder_input]
#      
#     eclone = clone_model(model)
#     for layer in eclone.layers[:7]:
#         encoder_layers.append(layer)
#     
#     em = Sequential(name='encoder')
#     for layer in encoder_layers:
#         em.add(layer)
# 
#     em.compile(optimizer='adam', loss='mse')
#     encoder = em
#     
#     # Isolate decoder
#     dclone = clone_model(model)
#     decoder_input = InputLayer(input_shape=(4, 4, 16), name='decoder_input')
#     decoder_layers = [decoder_input]
#     for layer in dclone.layers[7:]:
#         decoder_layers.append(layer)
#     
#     dm = Sequential(name='decoder')
#     for layer in decoder_layers:
#         dm.add(layer)
#     
#     dm.compile(optimizer='adam', loss='mse')
#     decoder = dm
# 
#     return model, encoder, decoder
