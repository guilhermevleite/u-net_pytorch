# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from model import UNET
# import utils as Utils
# from dataset import CarvanaDataset
# import albumentations as A
# !pip install --upgrade --force-reinstall --no-deps albumentations
# from albumentations.pytorch.transforms import ToTensorV2

# New Imports
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
import utils as Utils


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

class Train():

    def __init__(self, train_dir, train_maskdir, val_dir, val_maskdir, batch_size, n_epochs, n_workers, learning_rate, img_height, img_width, device, model, loss_fn):
        self.train_dir = train_dir
        print('constructor:', type(train_dir))
        print('self constructor:', type(self.train_dir))
        self.train_maskdir = train_maskdir
        self.val_dir = val_dir
        self.val_maskdir = val_maskdir
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_workers = n_workers
        self.learning_rate = learning_rate
        self.img_height = img_height
        self.img_width = img_width
        self.device = device
        self.model = model.to(device=self.device)
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
                )

        # TODO: If needed change Normalize mean to 0.0, and std to 1.0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)),
            transforms.Resize((
                self.img_height,
                self.img_width))
            ])

        self.train_loader, self.v_loader = Utils.get_loaders(
                train_dir=self.train_dir,
                train_maskdir=self.train_maskdir,
                val_dir=self.val_dir,
                val_maskdir=self.val_maskdir,
                batch_size=self.batch_size,
                train_transform=self.transform,
                val_transform=self.transform,
                num_workers=self.n_workers,
                pin_memory=True)


    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i%1000 == 999:
                last_loss = running_loss/1000
                print('\tBatch {} Loss: {}'.format(i+1, last_loss))
                running_loss = 0.0

        return last_loss


    def training(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1_000_000.0

        ep = int(self.n_epochs)

        for epoch in range(ep):
            print('Epoch:', epoch+1)

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)
            self.model.train(False)

            running_vloss = 0.0
            i = 0
            for i,v_data in enumerate(self.v_loader):
                v_inputs, v_labels = v_data
                v_outputs = self.model(v_inputs)
                v_loss = self.loss_fn(v_outputs, v_labels)

                running_vloss += v_loss

            avg_vloss = running_vloss / (i+1)
            print('Loss Train: {}\t Validation: {}'.format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
