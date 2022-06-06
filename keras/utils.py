import numpy as np
import os

allFileNames = os.listdir('/content/drive/MyDrive/db/segmentation/carvana/ori_images/')

np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])

train_FileNames = [ src + '/' + name for name in train_FileNames.tolist() ]
val_FileNames = [ src + '/' + name for name in val_FileNames.tolist() ]
test_FileNames = [ src + '/' + name for name in test_FileNames.tolist() ]

np.save(root_dir + 'names_train.npy', train_FileNames)
np.save(root_dir + 'names_val.npy', val_FileNames)
np.save(root_dir + 'names_test.npy', test_FileNames)
