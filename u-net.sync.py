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

# # U-Net

from google.colab import drive
drive.mount('/content/drive')

# +
import torch
import drive.MyDrive.segmentation.unet.lib.train as Train


class Spheroid():

    def __init__(self):
        learning_rate = 1e-4
        batch_size = 32
        num_epochs = 5
        num_workers = 2
        image_width = 240
        image_height = 160
        pin_memory = True
        load_model = False

        train_img_dir = '/content/drive/MyDrive/db/segmentation/FL5C/train/images/'
        train_mask_dir = '/content/drive/MyDrive/db/segmentation/FL5C/train/masks/'
        val_img_dir = '/content/drive/MyDrive/db/segmentation/FL5C/val/images/'
        val_mask_dir = '/content/drive/MyDrive/db/segmentation/FL5C/val/masks/'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def main():
        data_loader, trained_model = Train.main()
# -

Spheroid.main()
