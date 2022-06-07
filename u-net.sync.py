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

# + [markdown] id="80218602"
# # Requirements

# + colab={"base_uri": "https://localhost:8080/", "height": 264} id="771cbbc9" outputId="ace44c70-dccd-4923-b501-83c4d8fb9e2c"
# Install albumentations, with qudida
# TODO: Find a way to not use albumentations at all
# # !pip install --upgrade --force-reinstall --no-deps qudida==0.0.4
# # !pip install --upgrade --force-reinstall --no-deps albumentations==1.1.0


# IF cv2 is not working:
import cv2
if cv2.__version__ != '4.5.2':
  # !pip uninstall --yes opencv-python-headless==4.5.5.64
  # !pip install opencv-python-headless==4.5.2.52

# + colab={"base_uri": "https://localhost:8080/"} id="b7af6788" outputId="897990e4-a75d-485a-ccf4-1840ab31398c"
from google.colab import drive
drive.mount('/content/drive')

# + id="1fed527a"
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF
from torchsummary import summary
import matplotlib.pyplot as plt


# + [markdown] id="6d62ef33"
# # Dataset

# + id="84564df3"
class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, m_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.m_transform = m_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    # My adaptation
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # print('Getting:\n', img_path, '\n', mask_path)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.m_transform(mask)
        
        return image, mask
    
    # Video method
    # def __getitem__(self, index):
    #   img_path = os.path.join(self.image_dir, self.images[index])
    #   mask_path = os.path.join(self.mask_dir, self.images[index])

    #   image = np.array(Image.open(img_path).convert('RGB'))
    #   mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

    #   mask[mask == 255.0] = 1.
    #   image = self.transform(image)
    #   mask = self.m_transform(mask)

    #   return image, mask


# + [markdown] id="085ec470"
# # Utils

# + cellView="code" id="R6NyDvcH92kx"
#@title Default title text
if not os.path.isdir('saved_images'):
  # !mkdir saved_images
  
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
            # y = y.to(device).unsqueeze(1)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    
    dice = dice_score.item()/len(loader)
    acc = num_correct / num_pixels * 100

    # print('Batch results:')
    # print('\tDice score:', dice_score.item()/len(loader))
    # print('\tAccuracy: {:.2f}'.format(num_correct / num_pixels * 100)) 
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    # print(f"Dice score: {dice_score/len(loader)}")

# Change mode back to training mode
    model.train()

    return dice, acc


def save_predictions_as_imgs(
        loader,
        model,
        folder="saved_images/",
        device="cuda"
        ):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # preds = model(x)
            print('preds', torch.max(preds))
            preds = preds * 255
            print('preds', torch.max(preds))
            # preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
        render_img(preds.to('cpu'))

    model.train()


def plot_predictions(loader):
    _, ax = plt.subplots(4, 2)
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Target")

    l_iter = iter(loader)
    img, label = l_iter.next()

    img = torch.permute(img, (0, 2, 3, 1))
    label = label.squeeze(1)

    print('Length', len(img[0])-1)
    for i in range(len(img[0])-1):
      if i > 3:
        break

      print(i)
      print(torch.max(img[i]))
      img[i] = img[i] * 255
      print(torch.max(img[i]))

      print(img[i].shape, label[i].shape)
      ax[i, 0].imshow(img[i])
      ax[i, 1].imshow(label[i], cmap='gray')
      ax[i, 0].axis("off")
      ax[i, 1].axis("off")
    plt.show()


def plot_lists(list_one, list_two=None):
  if list_two == None:
    plt.plot(list_one)
    plt.show()
    
  else:
    plt.plot(list_one, list_two)
    plt.show()


def render_img(img):
    # img[0] = img[0] * 255
    img = img[0].permute((1, 2, 0))
    img = img.numpy().astype(int)
    print(type(img), img.shape, img.max())
    cv2.imwrite('preview.png', img)
    # img = img[0].permute((1, 2, 0))

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # plt.show()



# + [markdown] id="99cb05ed"
# # Model

# + id="cf759dfa"
# DoubleConv are the two blue arrows in the u_net diagram, horizontally
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
    # This is the blue arrow in the u_net diagram
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

        # Downward path, red arrows in the u_net diagram
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upward path, green arrows in the u_net diagram
        # TODO: For a better result we should use Transpose Convolutions
        for feature in reversed(features):
            # First append is the UP
            self.ups.append(
                nn.ConvTranspose2d(
                    # x2 because of the skip connection.
                    feature*2,
                    # This is the output
                    feature,
                    # These two will double the height, width of the image
                    kernel_size=2,
                    stride=2
                )
            )
            # Second append are the TWO CONVS, the in channels has to be double, refer to u_net diagram
            self.ups.append(DoubleConv(feature*2, feature))
        
        # This is the horizontal path between downward and upward, features[-1] = 512
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # This is the very last conv, from 388,388,64 to 388,388 or as in the paper: 388,388,2. It does not change the size of the image, it only changes the channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Stores the outputs that will skip into the upward path
        skip_connections = []
        
        # All the down layers are store in self.downs because we did it in __init__, so now, just run trough them
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
            # Notice that we are not overwriting skip_connectionS <-- plural
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


# + [markdown] id="P0DAk97hv8UR"
# ## Sanity Check on the Model

# + colab={"base_uri": "https://localhost:8080/"} id="FKa5QhGfv269" outputId="5da48ca1-b6d0-4708-f340-54c4dafda888"
def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == '__main__':
  test()

# + [markdown] id="0ec3c64c"
# # Train

# + id="91217028"
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
        self.lists = []
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
                )

        # TODO: If needed change Normalize mean to 0.0, and std to 1.0
        self.t_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.5, 0.5, 0.5), 
            #     (0.5, 0.5, 0.5)
            #     ),
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
            # clear_output()
            
            # print('Train this epoch, train_loader index:', i, '|', len(self.train_loader))
            inputs, labels = data

            # TODO: This needs to be revised, it was done to fix the input shape into the expected one
            # inputs = inputs.permute(0, 3, 1, 2)
            # labels = labels.unsqueeze(1)
            # print('shape:', inputs.shape, labels.shape)

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
                last_loss = running_loss/10
                print('\tBatch {} Loss: {}'.format(i+1, last_loss))
                running_loss = 0.0

            # print('1.', Utils.get_gpu_mem())
            del(inputs)
            del(labels)
            torch.cuda.empty_cache()
            # print('2.', Utils.get_gpu_mem())

        return last_loss


    # Video method
    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        # data = next(iter(self.train_loader))
        for i, data in enumerate(self.train_loader):
            # clear_output()
            
            # print('Train this epoch, train_loader index:', i, '|', len(self.train_loader))
            inputs, labels = data

            # TODO: This needs to be revised, it was done to fix the input shape into the expected one
            # inputs = inputs.permute(0, 3, 1, 2)
            # labels = labels.unsqueeze(1)
            # print('shape:', inputs.shape, labels.shape)

            inputs = inputs.float().to(self.device)
            labels = labels.float().to(self.device) # Just making sure

            with torch.cuda.amp.autocast():
              outputs = self.model(inputs)
              loss = self.loss_fn(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i%10 == 9 or i == 0:
                last_loss = running_loss/10
                print('\tBatch {}|{} Loss: {}'.format(i+1, len(self.train_loader), last_loss))
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
        best_dice = 0.0
        best_dice_ep = -1

        # early stop
        last_loss = 100
        patience = 3
        trigger_times = 0
        delta = 0.001

        # plotting
        v_dice_list = []
        dice_list = []
        loss = []

        for epoch in range(self.n_epochs):

            print('Epoch: {}|{}'.format(epoch+1, self.n_epochs))
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)
            self.model.train(False)

            # print('Validation for', epoch+1)
            # running_vloss = 0.0
            # i = 0
            # for i,v_data in enumerate(self.v_loader):
            #     v_inputs, v_labels = v_data
            #     # print('Input:', v_inputs.shape, 'label:', v_labels.shape)

            #     v_inputs = v_inputs.float().to(self.device)
            #     v_labels = v_labels.float().to(self.device)
            #     # print('Before dataloader:', Utils.get_gpu_mem())
            #     # v_inputs = v_inputs.permute(0, 3, 1, 2).float().to(self.device)
            #     # v_labels = v_labels.unsqueeze(1).float().to(self.device)
            #     # print('After dataloader:', Utils.get_gpu_mem())

            #     # print('Getting predictions')
            #     torch.cuda.empty_cache()
            #     # print('Before model:', Utils.get_gpu_mem())
            #     # with torch.cuda.amp.autocast():
            #     v_outputs = torch.sigmoid(self.model(v_inputs))

            #     # print('After model:', Utils.get_gpu_mem())
            #     # print(v_outputs.shape, v_labels.shape)
            #     v_loss = self.loss_fn(v_outputs, v_labels)                

            #     # v_outputs = v_outputs * 255
            #     # print('\tInput shape', v_inputs.shape)
            #     # print('\tInput Min Max', v_inputs.min(), v_inputs.max())
            #     # print('\tOutput Min Max', v_outputs.min(), v_outputs.max())
            #     # print('\tOutput shape', v_outputs.shape)
            #     # torchvision.utils.save_image(v_outputs, 'saved_images/v_output_epoch_{:03d}.png'.format(epoch+1))

            #     # print('3.', Utils.get_gpu_mem())
            #     del(v_inputs)
            #     del(v_labels)
            #     torch.cuda.empty_cache()
            #     # print('4.', Utils.get_gpu_mem())

            #     running_vloss += v_loss

            # # print('Calculating, average loss')
            # avg_vloss = running_vloss / (i+1)
            # print('Loss Train: {}\t Validation: {}'.format(avg_loss, avg_vloss))
            # print('---TRAINING---')
            dice, acc = check_accuracy(self.train_loader, self.model, device=self.device)
            # print('----------------')

            # print('---VALIDATION---')
            v_dice, v_acc = check_accuracy(self.v_loader, self.model, device=self.device)
            # print('----------------')
            print('\tdice: {:.5f} acc: {:.2f}\t v_dice: {:.5f} v_acc: {:.2f} E_S: {}'.format(dice, acc, v_dice, v_acc, trigger_times))

            # Early stopping
            if v_dice - delta <= last_loss:
              trigger_times += 1

              if trigger_times >= patience and epoch > 30:
                print('\n---Early Stop trigger---\n{} Epoch\t Dice: {}'.format(epoch+1, v_dice))
                return self.model

            else:
              trigger_times = 0

            last_loss = v_dice
            # END Early stopping

            # plotting
            loss.append(avg_loss)
            v_dice_list.append(v_dice)
            dice_list.append(dice)
            # END plotting

            # if dice > best_dice:
            #   best_dice = dice
            #   best_dice_ep = epoch_number

            # if avg_vloss < best_vloss:
            #     best_vloss = avg_vloss
            #     model_path = 'model_{}_{:05d}'.format(timestamp, epoch_number)
            #     torch.save(self.model.state_dict(), model_path)

            # print('Best dice so far: {}, epoch: {}'.format(best_dice, best_dice_ep))

        self.lists.append(loss)
        self.lists.append(dice_list)
        self.lists.append(v_dice_list)

        return self.model

# + [markdown] id="79d52b32"
# # U-Net

# + colab={"base_uri": "https://localhost:8080/"} id="c3596e44" outputId="425e378e-5e50-4568-f53a-bde6b9e90fa9"
local = False
path_suffix = None

if local:
    path_suffix = '/home/leite/Drive/'
else:
    path_suffix = '/content/drive/MyDrive/'

print('Suffix:', path_suffix)

train_dir = path_suffix + 'db/segmentation/mini-carvana/train/images'
train_maskdir = path_suffix + 'db/segmentation/mini-carvana/train/masks/'
val_dir = path_suffix + 'db/segmentation/mini-carvana/val/images/'
val_maskdir = path_suffix + 'db/segmentation/mini-carvana/val/masks/'

# This eliminates the need of SIGMOID
l_func = nn.BCEWithLogitsLoss()

print('Instantiating U-Net Traning!')
# TODO: Define optimizer out here
unet_train = Train(
        train_dir=train_dir,
        train_maskdir=train_maskdir,
        val_dir=val_dir,
        val_maskdir=val_maskdir,
        batch_size=32,
        n_epochs=1000,
        n_workers=2,
        learning_rate=1e-4,
        img_height=160,
        img_width=240,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model=UNET(in_channels=3, out_channels=1),
        loss_fn=l_func
        )

# summary(unet_train, (3, 160, 240))

# + colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="xVRuNSAGMdrA" outputId="41a0b133-f1ea-4d4a-a506-dfb363328d3e"
# print('Training U-Net... with', unet_train.device)
trained_model = unet_train.training()

# trained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    # in_channels=3, out_channels=1, init_features=32, pretrained=True)

# + id="0ddb2bed"
# save_predictions_as_imgs(unet_train.v_loader, unet_train.model)

# + id="hFMHodYy80bC"
# plot_predictions(unet_train.v_loader)

i_loader = iter(unet_train.v_loader)
img, lbl = i_loader.next()

print('Infos:')
print('\tType:', type(img[0]), type(lbl[0]))
print('\tShape:', img[0].shape, lbl[0].shape)
print('\tMax:', torch.max(img[0]), torch.max(lbl[0]))
fig = img[0].numpy()
print('\tMax:', fig.max())
fig = (fig*255).astype(int)
print('\tMax:', fig.max())

render_img(img)

# + id="2IIkjmZhEVYG"
plot_lists(unet_train.lists[1], unet_train.lists[2])

# + id="0lsR_ZArE7dZ"
plot_lists(unet_train.lists[0])
