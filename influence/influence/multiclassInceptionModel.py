from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import IPython
import tensorflow as tf
import math

from .hessians import hessians
from .genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from .inception_v3 import InceptionV3
from .inceptionModel import BinaryInceptionModel

from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.utils.data_utils import get_file

from keras import backend as K

class MulticlassInceptionModel(BinaryInceptionModel):

    def __init__(self, img_side, num_channels, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        
        self.img_side = img_side
        self.num_channels = num_channels
        self.input_dim = img_side * img_side * num_channels
        self.num_features = 2048 # Hardcoded for inception. For some reason Flatten() doesn't register num_features.

        super(BinaryInceptionModel, self).__init__(**kwargs)

        self.load_inception_weights()

        self.set_params_op = self.set_params()

        C = 1.0 / ((self.num_train_examples) * self.weight_decay)
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,
            max_iter=1000)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,
            max_iter=1000)        


    def inference(self, input):        
        reshaped_input = tf.reshape(input, [-1, self.img_side, self.img_side, self.num_channels])
        self.inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=reshaped_input)
        
        raw_inception_features = self.inception_model.output

        pooled_inception_features = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(raw_inception_features)
        self.inception_features = Flatten(name='flatten')(pooled_inception_features)

        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights', 
                [self.num_features * self.num_classes],
                stddev=1.0 / math.sqrt(float(self.num_features)),
                wd=self.weight_decay)            
            
            logits = tf.matmul(self.inception_features, tf.reshape(weights, [self.num_features, self.num_classes])) 
            
        self.weights = weights

        return logits


    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.num_features * self.num_classes],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)    
        return [set_weights]











