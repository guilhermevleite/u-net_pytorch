import os
import numpy as np
import cv2
from time import time
# import pickle

import keras

from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Dense, Cropping2D, Dropout, Concatenate
from keras.models import Model, load_model
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

output = Conv2D(3, 1, padding='same', activation='softmax')(layers[-1])

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(input=u_input, output=output)

model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=[keras_metrics.f1_score()])

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
print('Image shape:', train_images.shape)

# train_labels = []
# for file in os.listdir(label_path):
    # img = cv2.imread(label_path + file, 0)
    # img = cv2.resize(img, (388, 388), interpolation=cv2.INTER_NEAREST)
    # img = img/255
    # train_labels.append(img)
    # print(np.unique(img))

# train_generator = zip(train_image_generator, train_label_generator)

# print(train_label_generator)

# for it in train_label_generator:
    # print()

# results = model.fit_generator(train_generator, steps_per_epoch=10, epochs=3, verbose=2, callbacks=[tensorboard_callback])

# with open('weights.pkl', 'wb') as file:
    # pickle.dump(results.history, file)

model = load_model("model.h5", custom_objects={'dice_coef': dice_coef})

final = model.predict(x=train_images, batch_size=1, verbose=2)

idx, height, width, _ = train_images.shape
print('INPUT SHAPE:', train_images.shape)

print(final[0, :, :, :])
for n in range(idx):

    out = np.zeros((388, 388), dtype='uint8')
    for i in range(388):
        for j in range(388):
            # print('output:', train_images[n, i, j, 0])
            if final[n, i, j, 0] >= final[n, i, j, 1]:
                out[i, j] = 0
            else:
                out[i, j] = 255

    # print(out[0, 0])
    # print(np.unique(out))
    # print(out)

    cv2.imwrite('infers/'+str(n)+'.png', out)

print(final[0, 0, 0, :])
