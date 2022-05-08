import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv


# TODO: Look at Dataset class, why are we implementing __len__ and __getitem__, what else could we need to implement?
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", ".png"))
        # mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Since we are using sigmoid activation, it needs to be normalized between 0.0 and 1.0
        mask[mask == 255.0] = 1.0

        # Applying data augmentations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            # Getting the images back from the dictionary
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        # print('DEBUG:', self.images[index], image.shape, mask.shape)
        return image, mask