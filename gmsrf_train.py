from tqdm import tqdm
import torch.nn.functional as F
from operator import add
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from model import EncoderBlock
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

import math
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lib.utils_hrnet import shuffling, make_channel_first, make_channel_last, create_dir, epoch_time, print_and_save


def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".jpg" for name in data]
    masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class KvasirDataset(Dataset):
    """ Dataset for the Kvasir-SEG dataset. """
    def __init__(self, images_path, masks_path, size):
        """
        Arguments:
            images_path: A list of path of the images.
            masks_path: A list of path of the masks.
        """

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.n_samples = len(images_path)
    def __getitem__(self, index):
        """ Reading image and mask. """

        #image = Image.open(self.images_path[index])
        #mask = Image.open(self.masks_path[index])
        #image = image.resize((256,256))
        #mask = mask.resize((256,256))
        #mask = mask.convert('L')
        #image = skimage.io.imread(self.images_path[index],plugin="tifffile")
        #mask = skimage.io.imread(self.masks_path[index],plugin="tifffile")

        #mask = rgb2gray(mask)
        """ Resizing. """
        #image = cv2.resize(image, self.size)
        #mask = cv2.resize(mask, self.size)
        image = Image.open(self.images_path[index])
        mask = Image.open(self.masks_path[index])
        image = image.resize((256,256))
        mask = mask.resize((256,256))
        mask = mask.convert('L')
        """ Proper channel formatting. """
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        """ Normalization. """
        image = image/255.0
        mask = mask/255.0

        """ Changing datatype to float32. """
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        """ Changing numpy to tensor. """
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
    def __len__(self):
        return self.n_samples


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch,device):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
            # ---- data prepare ----
        images, gts = pack
        images = images.to(device)
        gts = gts.to(device)
            # ---- rescale ----
        trainsize = 256
            # ---- forward ----
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
        loss5 = structure_loss(lateral_map_5, gts)
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()
            # ---- recording loss ----
        # ---- train visualization ----
    save_path = 'files/'
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'GMSRF-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'GMSRF-%d.pth'% epoch)

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            y_pred,_,_,_ = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss


import os
import time
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from loss import DiceLoss, DiceBCELoss


def train_main(model):
    """ Seeding """

    """ Directories """

    """ Training logfile """
    train_log = open("files/train_log.txt", "w")


    train_list = glob('data/cvc_data/train/image/*.tif')
    train_mask_list = glob('data/cvc_data/train/mask/*.tif')
    """ Load dataset """
    train_x = train_list
    train_y = train_mask_list
    print(len(train_x),len(train_y))

    valid_x = glob('data/cvc_data/valid/image/*.tif')
    valid_y = glob('data/cvc_data/valid/mask/*.tif')

    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log, data_str)

    """ Hyperparameters """
    size = (256,256)
    batch_size = 6
    num_epochs = 130
    lr = 1e-4
    checkpoint_path = "checkpoint.pth"

    """ Dataset and loader """
    train_dataset = KvasirDataset(train_x, train_y, size)
    valid_dataset = KvasirDataset(valid_x, valid_y, size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


 
    """ Model """
    device = torch.device('cuda')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()
        
        train(train_loader, model, optimizer,epoch, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        print_and_save(train_log, data_str)
model = EncoderBlock()
train_main(model)
