#Model Rayyan Edit
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights,input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
###########################################################################
    conv21 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1),kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2),kernel_initializer = 'he_normal')(conv21)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv2)
    x2 = add([conv21,conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
###########################################################################
    conv31 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1),kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2),kernel_initializer = 'he_normal')(conv31)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv3)
    x3 = add([conv31,conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
###########################################################################
    conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv41)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    drop4 = (BatchNormalization())(conv4)
    x4 = add([drop4,conv41])
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
############################################################################
    conv51 = Conv2D(1024, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv51)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)
    drop5 = (BatchNormalization())(conv5)
    x5 = add([drop5,conv51])
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x5))
############################################################################
    merge6 = concatenate([x4,up6], axis = 3)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv61)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv6)
    x6 = add([conv61,conv6])
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x6))
############################################################################
    merge7 = concatenate([x3,up7], axis = 3)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv71)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv7)
    x7 = add([conv7,conv71])
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x7))
############################################################################
    merge8 = concatenate([x2,up8], axis = 3)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv81)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv8)
    x8 = add([conv8,conv81])
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x8))
############################################################################
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#############################################################################    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(input = inputs, output = conv10)    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


