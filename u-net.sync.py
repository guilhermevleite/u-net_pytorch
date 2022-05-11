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
# %cd unet

# ## Pull

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
# !pip uninstall opencv-python-headless==4.5.5.64
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


# train_dir = "/content/drive/MyDrive/DB/FL5C/train/images/"
train_dir = '/home/leite/Drive/db/segmentation/FL5C/train/images/'
# train_maskdir = "/content/drive/MyDrive/DB/FL5C/train/masks/"
train_maskdir = '/home/leite/Drive/db/segmentation/FL5C/train/masks/'
# val_dir = "/content/drive/MyDrive/DB/FL5C/val/images/"
val_dir = '/home/leite/Drive/db/segmentation/FL5C/val/images/'
# val_maskdir = "/content/drive/MyDrive/DB/FL5C/val/masks/"
val_maskdir = '/home/leite/Drive/db/segmentation/FL5C/val/masks/'



# TODO: Define optimizer out here
unet_train = Train(
        train_dir=train_dir,
        train_maskdir=train_maskdir,
        val_dir=val_dir,
        val_maskdir=val_maskdir,
        batch_size=32,
        n_epochs=5,
        n_workers=2,
        learning_rate=1e-4,
        img_height=160,
        img_width=240,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model=UNET(in_channels=3, out_channels=1),
        loss_fn=nn.BCEWithLogitsLoss()
        )

unet_train.training()
