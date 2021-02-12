#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, width = 256, height= 256, channels = 3, latent_size=100,model_type = 'DCGAN'):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        
        if model_type=='simple':
            self.Discriminator = self.model()
            self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
            self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
        elif model_type=='DCGAN':
            self.Discriminator = self.dc_model()
            self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)
            self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        
        self.summary()

    def dc_model(self):
        input_layer = Input((self.W, self.H, self.C))
        down_1 = Convolution2D(64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(input_layer)

        down_2 = Convolution2D(64 * 2 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(down_1)
        norm_2 =  BatchNormalization()(down_2)

        down_3 = Convolution2D(64 * 4 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 =  BatchNormalization()(down_3)

        down_4 = Convolution2D(64 * 8 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 =  BatchNormalization()(down_4)

        down_5 = Convolution2D(64 * 8 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_4)
        norm_5 =  BatchNormalization()(down_5)

        down_6 = Convolution2D(64 * 8 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_5)
        norm_6 =  BatchNormalization()(down_6)

        down_7 = Convolution2D(64 * 8 , kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(norm_6)
        norm_7 =  BatchNormalization()(down_7)

        flat = Flatten()(norm_7)
        output_layer = Dense(1, activation='sigmoid')(flat)


        model = Model(input_layer, output_layer)
        return model

    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()


if __name__ == "__main__":
    
    model = Discriminator()