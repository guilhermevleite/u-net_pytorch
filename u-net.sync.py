# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Github stuff

# ## Clone

# !git clone https://github.com/guilhermevleite/u-net_pytorch unet
# %cd /content/unet

# ## Pull

# %cd /content/unet
# !git pull

# ## Commit changes

# !git add . 
# !git commit -m 'changes made in colab'
# !git push

# # Requirements

# +
# Install albumentations, with qudida
# TODO: Find a way to not use albumentations at all
# !pip install --upgrade --force-reinstall --no-deps qudida==0.0.4
# !pip install --upgrade --force-reinstall --no-deps albumentations==1.1.0

# IF cv2 is not working:
# !pip uninstall --yes opencv-python-headless==4.5.5.64
# !pip install opencv-python-headless==4.5.2.52
# -

# # U-Net

from google.colab import drive
drive.mount('/content/drive')

# +
import torch
import torch.nn as nn
import train as Train
from model import UNET

local = True
path_suffix = None

if local:
    path_suffix = '/home/leite/Drive/'
else:
    path_suffix = '/content/drive/MyDrive/'

print('Suffix:', path_suffix)

train_dir = path_suffix + 'db/segmentation/FL5C/train/images/'
train_maskdir = path_suffix + 'db/segmentation/FL5C/train/masks/'
val_dir = path_suffix + 'db/segmentation/FL5C/val/images/'
val_maskdir = path_suffix + 'db/segmentation/FL5C/val/masks/'

l_func = nn.BCEWithLogitsLoss()

print('Instantiating U-Net Traning!')
# TODO: Define optimizer out here
unet_train = Train.Train(
        train_dir=train_dir,
        train_maskdir=train_maskdir,
        val_dir=val_dir,
        val_maskdir=val_maskdir,
        batch_size=2,
        n_epochs=5,
        n_workers=2,
        learning_rate=1e-4,
        img_height=160,
        img_width=240,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model=UNET(in_channels=3, out_channels=1),
        loss_fn=l_func
        )

print('Training U-Net... with', unet_train.device)
unet_train.training()
# -
print('Done Training.')
print('Haha')


