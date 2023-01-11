# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:57:06 2019

@author: rayyan
"""
import os
import random
import re
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from keras import backend as K
from scipy import ndimage
import natsort
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
###############################################################################
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=40,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   zca_whitening=False)
        
val_datagen = ImageDataGenerator(rescale=1./255)
  
def trainGenerator(image_path,mask_path,size,batch):
    train_image_generator = train_datagen.flow_from_directory(image_path,
                                                         batch_size = batch,
                                                         class_mode= None,
                                                         color_mode= 'rgb',
                                                         seed = 1,
                                                         target_size=size)
    train_mask_generator = train_datagen.flow_from_directory(mask_path,
                                                         batch_size = batch,
                                                         class_mode= None,
                                                         color_mode= 'grayscale',
                                                         seed = 1,
                                                         target_size=size)
    train_generator = zip(train_image_generator, train_mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)  

def valGenerator(image_path,mask_path,size,batch):
    val_image_generator = val_datagen.flow_from_directory(image_path,
                                                      batch_size = batch,
                                                      class_mode= None,
                                                      color_mode= 'rgb',
                                                      seed = 1,
                                                      target_size=size)
    val_mask_generator = val_datagen.flow_from_directory(mask_path,
                                                     batch_size = batch,
                                                     class_mode= None,
                                                     color_mode= 'grayscale',
                                                     seed = 1,
                                                     target_size=size)
    val_generator = zip(val_image_generator, val_mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)


def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img/255
        mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)


def testGenerator(test_path,target_size):
    ls=natsort.natsorted(os.listdir(test_path))
    for image_file_name in ls:
        if image_file_name.endswith(".tif"):
            img = cv2.imread(test_path+image_file_name)
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img=cv2.resize(img,target_size)
            img = np.reshape(img,(1,)+img.shape)
            yield img



def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d.tif"%i),img)

def dice_loss():
    def DICE(y_true,y_pred):
        #y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        #y_true = tf.nn.sigmoid(y_true)
        #y_pred = tf.nn.sigmoid(y_pred)
        #y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        #y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        #y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        # intersection = tf.reduce_sum(y_true_f * y_pred_f)
        # union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        # dice = (2. * intersection) / (union + 0.00001)
        return (1-(2*K.sum(y_true*y_pred)/(K.sum(y_true) + K.sum(y_pred))))
    return DICE
"""
def dice_loss():
    def DICE(y_true,y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.nn.sigmoid(y_pred)
        #y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice = (2. * intersection) / (union + 0.00001)
        return dice         #keras.mean(2*keras.sum(y_true*y_pred)/(keras.sum(y_true) + keras.sum(y_pred)))
    return DICE
"""
def rvd_loss():
    def RAVD(y_true,y_pred):
        #y_true_f = tf.nn.sigmoid(y_true)
        #y_true_f = tf.cast(tf.reshape(y_true_f, [-1]), tf.float32)
        #y_pred_f = tf.nn.sigmoid(y_pred)
        #y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
        #y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)
        rvd= tf.abs(K.sum(tf.cast(y_true,tf.float32) - tf.cast(y_pred,tf.float32))/(K.sum(tf.cast(y_true,tf.float32)+1)))
        return rvd
    return RAVD
def com_loss():
    def COM(y_true,y_pred):
        rvd=tf.abs(K.sum(tf.cast(y_true,tf.float32) - tf.cast(y_pred,tf.float32))/(K.sum(tf.cast(y_true,tf.float32)+1)))
        dice=(1-(2*K.sum(y_true*y_pred)/(K.sum(y_true) + K.sum(y_pred))))
        com=(rvd+dice)
        return com
    return COM
