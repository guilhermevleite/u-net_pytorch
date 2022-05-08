from time import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Dense, Cropping2D, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras_metrics

from keras.callbacks import TensorBoard

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

u_input = Input((572, 572, 3))

layers = []
# First downward level
layers.append(Conv2D(64, 3, activation='relu')(u_input))
layers.append(Conv2D(64, 3, activation='relu')(layers[-1]))
skip_4 = layers[-1] # 4th merge connection
layers.append(MaxPooling2D(2, strides=2)(layers[-1]))
layers.append(Dropout(0.1)(layers[-1]))

# Second downward level
layers.append(Conv2D(128, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(128, 3, activation='relu')(layers[-1]))
skip_3 = layers[-1] # 3th merge connection
layers.append(MaxPooling2D(2, strides=2)(layers[-1]))
layers.append(Dropout(0.1)(layers[-1]))

# Third downward level
layers.append(Conv2D(256, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(256, 3, activation='relu')(layers[-1]))
skip_2 = layers[-1] # 2nd merge connection
layers.append(MaxPooling2D(2, strides=2)(layers[-1]))
layers.append(Dropout(0.1)(layers[-1]))

# Fourth downward level
layers.append(Conv2D(512, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(512, 3, activation='relu')(layers[-1]))
skip_1 = layers[-1] # 1st merge connection
layers.append(MaxPooling2D(2, strides=2)(layers[-1]))
layers.append(Dropout(0.1)(layers[-1]))

# Fifth downward level
layers.append(Conv2D(1024, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(1024, 3, activation='relu')(layers[-1]))

# First upward level
crop = Cropping2D((4, 4))(skip_1)
layers.append(UpSampling2D(2)(layers[-1]))
layers.append(Conv2D(512, 2, padding='same', activation='relu')(layers[-1]))
layers.append(Concatenate(axis=3)([layers[-1], crop]))
layers.append(Dropout(0.1)(layers[-1]))
layers.append(Conv2D(512, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(512, 3, activation='relu')(layers[-1]))

# Second upward level
crop = Cropping2D((16, 16))(skip_2)
layers.append(UpSampling2D(2)(layers[-1]))
layers.append(Conv2D(256, 2, padding='same', activation='relu')(layers[-1]))
layers.append(Concatenate(axis=3)([layers[-1], crop]))
layers.append(Dropout(0.1)(layers[-1]))
layers.append(Conv2D(256, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(256, 3, activation='relu')(layers[-1]))

# Third upward level
crop = Cropping2D((40, 40))(skip_3)
layers.append(UpSampling2D(2)(layers[-1]))
layers.append(Conv2D(128, 2, padding='same', activation='relu')(layers[-1]))
layers.append(Concatenate(axis=3)([layers[-1], crop]))
layers.append(Dropout(0.1)(layers[-1]))
layers.append(Conv2D(128, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(128, 3, activation='relu')(layers[-1]))

# Fourth upward level
crop = Cropping2D((88, 88))(skip_4)
layers.append(UpSampling2D(2)(layers[-1]))
layers.append(Conv2D(64, 2, padding='same', activation='relu')(layers[-1]))
layers.append(Concatenate(axis=3)([layers[-1], crop]))
layers.append(Dropout(0.1)(layers[-1]))
layers.append(Conv2D(64, 3, activation='relu')(layers[-1]))
layers.append(Conv2D(64, 3, activation='relu')(layers[-1]))

output = Conv2D(2, 1, padding='same', activation='softmax')(layers[-1])

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(input=u_input, output=output)

model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=[dice_coef])

# tensorboard_callback = TensorBoard(log_dir='logs/')

print(model.summary())

# train_path = '/home/leite/Workspace/datasets/membrane/train/image/'
train_path = '/mnt/Data/leite/membrane/train/image/'
# label_path = '/home/leite/Workspace/datasets/membrane/train/label/'
label_path = '/mnt/Data/leite/membrane/train/label/'

train_images = []
for file in os.listdir(train_path):
    img = cv2.imread(train_path + file)
    img = cv2.resize(img, (572,  572))
    img = img/255
    train_images.append(img)

train_images = np.asarray(train_images)
# print('Image shape:', train_images.shape)

train_labels = []
for file in os.listdir(label_path):
    img = cv2.imread(label_path + file, 0)
    img = cv2.resize(img, (572, 572), interpolation=cv2.INTER_NEAREST)
    img = img[92:480, 92:480]
    img = img/255
    train_labels.append(img)
    # print(np.unique(img))

# cv2.imshow('Some', train_labels[-1])
# cv2.waitKey(0)
# plt.imshow(train_labels[-1])
# plt.show()

train_labels = np.asarray(train_labels)
train_labels = np.expand_dims(train_labels, axis=4)
# print('Label shape:', train_labels.shape)

train_generator = zip(train_images, train_labels)
# print('Generator:', type(train_generator))

# results = model.fit_generator(train_generator, steps_per_epoch=30/15, epochs=2, verbose=2, callbacks=[tensorboard_callback])
model.fit(x=train_images, y=train_labels, batch_size=2, epochs=50, class_weight=[1, 10], validation_split=0.1666, verbose=2)

model.save('model.h5')

# print(results)

# final = model.predict_generator(train_generator, steps=3, callbacks=[tensorboard_callback], verbose=2)
# print(final)
