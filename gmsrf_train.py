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
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)

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


    train_list = glob('../../data/cvc_data/train/image/*.tif')
    train_mask_list = glob('../../data/cvc_data/train/mask/*.tif')
    """ Load dataset """
    train_x = train_list
    train_y = train_mask_list
    print(len(train_x),len(train_y))

    valid_x = glob('../../data/cvc_data/valid/image/*.tif')
    valid_y = glob('../../data/cvc_data/valid/mask/*.tif')

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


    model.load_state_dict(torch.load('files/PraNet-19.pth', map_location='cpu'))
    """ Model """
    device = torch.device('cuda')
    #model = CompNet()
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
from sklearn.metrics import f1_score, recall_score, precision_score,jaccard_score
import statistics
import time
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    dice_all = []
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            dice_all.append(channel_dice)
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    std = statistics.stdev(dice_all)
    return mean_dice_channel,std,np.asarray(dice_all)


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    ## Score
    score_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_recall = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_precision = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary', zero_division=0)
    score_iou = jaccard_score(y_true.reshape(-1),y_pred.reshape(-1))
    return [score_f1, score_recall, score_precision,score_iou]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask
def main_test():
    """ Seeding """
    create_dir("results")
    test_img_list = glob("../../data/test_img/images/*.jpg")
    test_mask_list = glob("../../data/test_img_m/masks/*.jpg")
    test_img_list = glob("../../data/cvc_data/test/image/*.tif")
    test_mask_list = glob("../../data/cvc_data/test/mask/*.tif")
    test_x, test_y = test_img_list,test_mask_list

    # """ CVC-ClinicDB """
    # test_x = sorted(glob("/media/nikhil/ML/ml_dataset/CVC-612/images/*"))
    # test_y = sorted(glob("/media/nikhil/ML/ml_dataset/CVC-612/masks/*"))

    """ Hyperparameters """
    size = (256, 256)
    checkpoint_path = "files/PraNet-79.pth"

    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PraNet()
    print('no of params:',get_n_params(model))
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Testing """
    pred_all = []
    mask_all = []
    dice_all = []
    """ Testing """
    metrics_score = [0.0, 0.0, 0.0, 0.0]
    dice_all_images,iou_all,recall_all,precision_all = [],[],[],[]
    total_time = 0

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        ## Image

        image = Image.open(x)
        image = image.resize((256,256))
        ori_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)
        
        ## Mask
        mask = Image.open(y)
        mask = mask.resize((256,256))
        mask = mask.convert('L')
        ori_mask = mask
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        #print(mask.size())
        with torch.no_grad():
            start_time = time.time()
            #pred_y = model(image)
            #pred_y = torch.sigmoid(pred_y)
            #print(pred_y.size())
            res5, res4, res3, res2 = model(image)
            res = res5
            res = F.upsample(res, size=256, mode='bilinear', align_corners=False)
            res = torch.sigmoid(res)
            pred_y = res
            ending_time = time.time() - start_time
            total_time = total_time - ending_time
            score = calculate_metrics(mask, pred_y)
            
            score = calculate_metrics(mask, pred_y)
            dice_all.append(score[0])
            dice_image = score[0]
            if math.isnan(dice_image):
                dice_image = 0
            iou_image = score[3]
            if math.isnan(iou_image):
                iou_image = 0
            recall_image = score[1]
            if math.isnan(recall_image):
                recall_image= 0
            precision_image = score[2]
            if math.isnan(precision_image):
                precision_image = 0


            dice_all_images.append(dice_image)
            iou_all.append(iou_image)
            recall_all.append(recall_image)
            precision_all.append(precision_image)


            metrics_score = list(map(add, metrics_score, score))
            ## Mask
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = pred_y * 255
            # pred_y = np.transpose(pred_y, (1, 0))
            pred_y = np.array(pred_y, dtype=np.uint8)
            pred_all.append(pred_y)
            mask_all.append(mask)
        ori_img     = ori_img
        ori_mask    = mask_parse(ori_mask)
        pred_y      = mask_parse(pred_y)
        sep_line    = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img, sep_line,
            ori_mask, sep_line,
            pred_y, sep_line,
        ]

        cat_images = np.concatenate(tmp, axis=1)
        #cv2.imwrite(f"results/{name}.png", cat_images)
    pred_all = np.asarray(pred_all)
    print(len(mask_all),mask_all[0].shape)
    dice_all = np.asarray(dice_all)
    new_masks = []
    for m in mask_all:
        print(m.shape)
        m = np.asarray(m.cpu())
        new_masks.append(m)
    new_masks = np.asarray(new_masks)
    pred_all = np.expand_dims(pred_all,axis=-1)
    _,std,_ = mean_dice_coef(pred_all,new_masks)
    print(dice_all)
    np.save('kvasir_pranet_dice.npy',dice_all)
    std_iou= statistics.stdev(iou_all)
    std_recall = statistics.stdev(recall_all)
    std_precision = statistics.stdev(precision_all)
    std = statistics.stdev(dice_all_images)
    f1 = metrics_score[0]/len(test_x)
    recall = metrics_score[1]/len(test_x)
    precision = metrics_score[2]/len(test_x)
    iou = metrics_score[3]/len(test_x)
    FPS = len(test_x) / total_time
    print(f"F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - IOU: {iou:1.4f} - FPS:{FPS:1.4f} - std:{std:1.4f}")
    print(f"F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - IOU:{iou:1.4f}")
    print(f"std of iou:{std_iou:1.4f} - Std of recall: {std_recall:1.4f} - std of precision: {std_precision:1.4f}")


main_test()
