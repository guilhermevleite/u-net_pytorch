import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from drive.MyDrive.libs.unet.model import UNET
import drive.MyDrive.libs.unet.utils as Utils
from drive.MyDrive.libs.unet.dataset import CarvanaDataset
import albumentations as A
# !pip install --upgrade --force-reinstall --no-deps albumentations
from albumentations.pytorch.transforms import ToTensorV2


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "/content/drive/MyDrive/DB/FL5C/train/images/"
TRAIN_MASK_DIR = "/content/drive/MyDrive/DB/FL5C/train/masks/"
VAL_IMG_DIR = "/content/drive/MyDrive/DB/FL5C/val/images/"
VAL_MASK_DIR = "/content/drive/MyDrive/DB/FL5C/val/masks/"


# Train function will do ONE epoch of training
def train_fn(loader, model, optimizer, loss_fn, scaler):
    # Progress bar
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        # Send to device
        data = data.to(device=DEVICE)
        # It should already be float, but for some reason they are rassuring it is float
        # Unsqueeze to add one channel dimension. Is this because the output is (N, 1, height, width) instead of (N, height, width)? Maybe!
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward will be float16 training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # Loss of predictions and targets
            loss = loss_fn(predictions, targets)

        # Forward without GPU
        # predictions = model(data)
        # Loss of predictions and targets
        # loss = loss_fn(predictions, targets)

        # Backward
        # Zero all the gradients from previous
        optimizer.zero_grad()
        # TODO: WTF are these scalers!?
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop, showing loss function so far.
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Normalize(
             mean=[0.0, 0.0, 0.0],
             std=[1.0, 1.0, 1.0],
             max_pixel_value=255.0
         ),
         ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Normalize(
             mean=[0.0, 0.0, 0.0],
             std=[1.0, 1.0, 1.0],
             max_pixel_value=255.0
         ),
         ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    # This will not work with multiple classes, it would need to be Cross Entropy Loss
    loss_fn = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = Utils.get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        Utils.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Save model
        # TODO: Fix save_checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        Utils.save_checkpoint(checkpoint)

        # Check accuracy
        Utils.check_accuracy(val_loader, model, device=DEVICE)

        # Save predictions to folder
        Utils.save_predictions_as_imgs(
            val_loader,
            model,
            folder="/content/drive/MyDrive/DB/uNet/infers/",
            device=DEVICE
        )
    
    return (train_loader, model)
    # plot_predictions(val_loader, model, DEVICE)