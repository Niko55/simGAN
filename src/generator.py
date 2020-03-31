from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
from loss import self_regularization_loss
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adamax
from tensorflow.keras.utils import plot_model

class Generator(object):
    def __init__(self, width = 35, height= 55, channels = 1 ,name='generator'):

        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (height ,width ,channels)
        self.NAME = name

        self.Generator = self.model()
        self.OPTIMIZER = SGD(lr=0.001)
        self.Generator.compile(loss=self_regularization_loss, optimizer=self.OPTIMIZER)

        self.save_model_graph()
        self.summary()

    def model(self):
        input_layer = tf.keras.Input(shape=self.SHAPE)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        output_layer = layers.Conv2D(self.C, (1, 1), activation='tanh')(x)
        return Model(input_layer, output_layer)

    def summary(self):
        return self.Generator.summary()

    def save_model_graph(self):
        plot_model(self.Generator, to_file='/out/Generator_Model.png')

    def save_model(self ,epoch ,batch):
        self.Generator.save('/out/ ' +self.NAME +'_Epoch_ ' +epoch +'_Batch_ ' +batch +'model.h5')