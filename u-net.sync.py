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

from google.colab import drive
drive.mount('/content/drive')

import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF

# # Dataset

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, m_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.m_transform = m_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.m_transform(mask)

        # mask[mask == 255.0] = 1.0
        
        return image, mask

# # Utils

# +
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        mask_transform,
        num_workers=0,
        pin_memory=True
        ):

    train_ds = MyDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        m_transform=mask_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = MyDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=train_transform,
        m_transform=mask_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def get_gpu_mem(index=0):
    r = torch.cuda.memory_reserved(index)
    a = torch.cuda.memory_allocated(index)

    free, total = torch.cuda.mem_get_info(index)
    free = (free * 8) / (8 * 1000 * 1000 * 1000)
    total = (total * 8) / (8 * 1000 * 1000 * 1000)

    # return r-a
    # return torch.cuda.mem_get_info(index)
    return '{:.02f}% Free'.format(free*100/total)


def save_checkpoint(
        state, 
        filename="my_checkpoint.pth.tar"
        ):

    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
# Change mode to evaluation and change back to training at the end of this function
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # TODO: What exactly is happening when I call something.to(device)?
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100.0:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

# Change mode back to training mode
    model.train()


def save_predictions_as_imgs(
        loader,
        model,
        folder="saved_images/",
        device="cuda"
        ):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
# -

# # Model

# +
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # TODO Figure out what is this doing here. I need its a reference to nn.Module, since we are inheriting that class.
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # First conv
            # Conv2d(in_cha, out_cha, kernel, stride, padding)
            # When we set stride and padding to one, it is called a SAME CONVOLUTION, the input height and width is the same after the convolution.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # There was no BachNorm at the time uNet was published, but it helps, so we are going to use it, and to do that, Conv2d 'bias' argument has to be False.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    # TODO: What the fuck is this doing? Isn't self.conv just initiated inside __init__?
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        # In the paper, the out channel was 2, we are going to use 1, since all we want is a binary segmentation.
        # TODO: Check whether out channel > 1 is necessary only when doing semantic segmentation.
        out_channels=1,
        # This as the features on every double convolution
        features=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        # We can not use self.downs = [], because it stores the convs and we want do do eval on these.
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upward path
        # TODO: For a better result we should use Transpose Convolutions
        for feature in reversed(features):
            # First append is the UP
            self.ups.append(
                nn.ConvTranspose2d(
                    # x2 because of the skip connection.
                    feature*2,
                    feature,
                    # These two will double the height, width of the image
                    kernel_size=2,
                    stride=2
                )
            )
            # Second append are the TWO CONVS
            self.ups.append(DoubleConv(feature*2, feature))
        
        # This is the horizontal path between downward and upward
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # This is the very last conv, from 388,388,64 to 388,388 or as in the paper: 388,388,2. It does not change the size of the image, it only changes the channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        # Simply reversing the list, because of the upward path will use it in inverse order
        skip_connections = skip_connections[::-1]
        # Step=2 because the upward path has a UP and a DoubleConv, but the skip only applies to the UP part.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # Integer division by 2 because, altough we want to skip the DoubleConv, we also want to run through the skip_connections one by one.
            skip_connection = skip_connections[idx//2]

            ''' The INPUT needs to be shaped on a multiple of 16, since it is four down ways. If that is not the case, there will be an error to concatenate because of the MAXPOOL, since them both need same height and width.
            One work around this is to check if they are different and resize the X '''
            if x.shape != skip_connection.shape:
                # Shape has: 0 BATCH_SIZE, 1 N_CHANNELS, 2 HEIGHT, 3 WIDTH. With [2:] we are taking only height and width.
                x = TF.resize(x, size=skip_connection.shape[2:])

            # We have 4 dims, 0 BATCH, 1 CHANNEL, 2 HEIGHT, 3 WIDTH. We are concatenating them along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # This will do the DoubleConv after we did the UP and concatenated the skip connection
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


# -

# # Train

# +
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
# import utils as Utils

from IPython.display import clear_output 


class Train():

    def __init__(self, train_dir, train_maskdir, val_dir, val_maskdir, batch_size, n_epochs, n_workers, learning_rate, img_height, img_width, device, model, loss_fn):
        self.train_dir = train_dir
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
        self.t_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)
                ),
            ])
        self.m_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            ])

        self.train_loader, self.v_loader = get_loaders(
                train_dir=self.train_dir,
                train_maskdir=self.train_maskdir,
                val_dir=self.val_dir,
                val_maskdir=self.val_maskdir,
                batch_size=self.batch_size,
                train_transform=self.t_transform,
                mask_transform=self.m_transform,
                num_workers=self.n_workers,
                pin_memory=True)


    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.train_loader):
            clear_output()
            
            print('Train One Epoch, train_loader index:', i, len(self.train_loader))
            inputs, labels = data

            # TODO: This needs to be revised, it was done to fix the input shape into the expected one
            # inputs = inputs.permute(0, 3, 1, 2)
            # labels = labels.unsqueeze(1)
            print('shape:', inputs.shape, labels.shape)

            inputs = inputs.float().to(self.device)
            labels = labels.float().to(self.device) # Just making sure

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
              outputs = self.model(inputs)
              loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i%10 == 9:
                last_loss = running_loss/1000
                print('\tBatch {} Loss: {}'.format(i+1, last_loss))
                running_loss = 0.0

            # print('1.', Utils.get_gpu_mem())
            del(inputs)
            del(labels)
            torch.cuda.empty_cache()
            # print('2.', Utils.get_gpu_mem())

        return last_loss


    def training(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1_000_000.0

        for epoch in range(self.n_epochs):
            print('Epoch:', epoch+1)

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)
            self.model.train(False)

            # print('Validation for', epoch+1)
            running_vloss = 0.0
            i = 0
            for i,v_data in enumerate(self.v_loader):
                v_inputs, v_labels = v_data
                print('Input:', v_inputs.shape, 'label:', v_labels.shape)

                # print('Before dataloader:', Utils.get_gpu_mem())
                v_inputs = v_inputs.permute(0, 3, 1, 2).float().to(self.device)
                v_labels = v_labels.unsqueeze(1).float().to(self.device)
                # print('After dataloader:', Utils.get_gpu_mem())

                # print('Getting predictions')
                torch.cuda.empty_cache()
                # print('Before model:', Utils.get_gpu_mem())
                # with torch.cuda.amp.autocast():
                v_outputs = self.model(v_inputs)
                # print('After model:', Utils.get_gpu_mem())
                # print(v_outputs.shape, v_labels.shape)
                v_loss = self.loss_fn(v_outputs, v_labels)                

                # print('3.', Utils.get_gpu_mem())
                del(v_inputs)
                del(v_labels)
                torch.cuda.empty_cache()
                # print('4.', Utils.get_gpu_mem())

                running_vloss += v_loss

            # print('Calculating, average loss')
            avg_vloss = running_vloss / (i+1)
            print('Loss Train: {}\t Validation: {}'.format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
# -

# # U-Net

# +
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
unet_train = Train(
        train_dir=train_dir,
        train_maskdir=train_maskdir,
        val_dir=val_dir,
        val_maskdir=val_maskdir,
        batch_size=32,
        n_epochs=3,
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
print('Done Training')
