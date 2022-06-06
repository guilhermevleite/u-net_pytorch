import os
import shutil
import random

source_folder = '/home/leite/Drive/db/segmentation/carvana/ori_images/'
train_folder = '/home/leite/Drive/db/segmentation/carvana/train/'
val_folder = '/home/leite/Drive/db/segmentation/carvana/val/'

print('Reading file list')
file_list = os.listdir(source_folder)
file_list.sort()

split_ratio = 20

val_num = int(len(file_list) * 20 / 100)
print('Size of validation:', val_num)

for idx, file in enumerate(file_list):
    print('Moving file', file, 'to:')
    if val_num > 0 and random.randint(0, 100) < split_ratio:
        print('\tValidation', val_num, file)
        shutil.copyfile(source_folder + file, val_folder + 'images/' + file)

        source_folder = source_folder.replace('ori_images', 'ori_masks')
        # file = file.replace('jpg', 'gif')

        shutil.copyfile(source_folder + file, val_folder + 'masks/' + file)
        val_num -= 1

    else:
        print('\tTraining', idx, file)
        shutil.copyfile(source_folder + file, train_folder + 'images/' + file)

        source_folder = source_folder.replace('ori_images', 'ori_masks')
        # file = file.replace('jpg', 'gif')

        shutil.copyfile(source_folder + file, train_folder + 'masks/' + file)

print(int(len(file_list) * 20 / 100))
