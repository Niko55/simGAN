from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
from loss import local_adversarial_loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adamax
from tensorflow.keras.utils import plot_model

class Discriminator(object):
    def __init__(self, width=35, height=55, channels=1, name='discriminator'):
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (height, width, channels)
        self.NAME = name
        self.Discriminator = self.model()
        self.OPTIMIZER = SGD(lr=0.001)
        self.Discriminator.compile(loss=local_adversarial_loss, optimizer=self.OPTIMIZER)
        self.save_model_graph()
        self.summary()

    def model(self):
        input_layer = tf.keras.Input(shape=self.SHAPE)
        x = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(input_layer)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((3, 3), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (1, 1), activation='relu')(x)
        x = layers.Conv2D(2, (1, 1), activation='relu')(x)
        output_layer = layers.Reshape(-1, 2)(x)
        return Model(input_layer, output_layer)

    def summary(self):
        return self.Discriminator.summary()

    def save_model_graph(self):
        plot_model(self.Discriminator, to_file='/data/Discriminator_Model.png')

    def save_model(self, epoch, batch):
        self.Discriminator.save('/out/ ' + self.NAME + '_Epoch_ ' + epoch + '_Batch_ ' + batch + 'model.h5')
