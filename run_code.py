# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:27:08 2018

@author: kr
"""

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import pandas as pd
from unet_model import UnetModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess_file import ProcessFile

# these snippets are used to invoke GPU if available or no of cpu cores
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

############################################################
# Creating CSV file of image pixels and mask pixels
loc_image = './data/images/'   # location of raw image
loc_poly = './data/polygon/'   # Location of polygon files

pfile = ProcessFile(loc_image, loc_poly, 1280)
processed_image_df, mask_df = pfile.get_custom_image_mask(size=[128, 128])

# save file to some location for feeding in neural network
processed_image_df.to_csv('./data/train/mask_file.csv', index=False)
mask_df.to_csv('./data/train/train_file.csv', index=False)

#############################################################

## reading data that has been processed to 128 pixel size and its mask
df_image = pd.read_csv('./processed_data/image_file.csv')
df_mask = pd.read_csv('./processed_data/mask_file.csv')

# scaling data
image_array=df_image.values.reshape([len(df_image),128,128,1])/255.
mask_array=df_mask.values.reshape([len(df_image),128,128,1])/255.

# spliting data for train and test
x_train, x_test, y_train, y_test = train_test_split(image_array, mask_array, test_size=0.1)

# define the size of image prepared through preprocessing
row_size = 128
col_size = 128

unet = UnetModel(row_size, col_size, 1)

model = unet.small_unet()
model_checkpoint = ModelCheckpoint('unet_bd.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(x_train, y_train,  batch_size=64, nb_epoch=10, verbose=1, shuffle=True,
validation_split=0.2, callbacks=[model_checkpoint])

res = model.predict(x_test)

xx = res[1,:].reshape([128,128])

i=2
plt.imshow((x_test[i,:]).reshape([128,128]))
plt.imshow((res[i,:]).reshape([128,128]))
plt.imshow(res[i,:].reshape([128,128])>res[i,:].reshape([128,128]).mean())



