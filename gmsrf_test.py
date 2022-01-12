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
import cv2

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
        
        image = Image.open(self.images_path[index])
        mask = Image.open(self.masks_path[index])
        image = image.resize((256,256))
        mask = mask.resize((256,256))
        mask = mask.convert('L')
        """ Resizing. """
        #image = cv2.resize(image, self.size)
        #mask = cv2.resize(mask, self.size)

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



import os
import time
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from loss import DiceLoss, DiceBCELoss


model = EncoderBlock()
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
def main_test(test_img_list,test_mask_list,dataset):
    """ Seeding """
    create_dir("results")
    #test_img_list = glob("../../data/test_img/images/*.jpg")
    #test_mask_list = glob("../../data/test_img_m/masks/*.jpg")
    #test_img_list = glob("../../data/cvc_data/test/image/*.tif")
    #test_mask_list = glob("../../data/cvc_data/test/mask/*.tif")
    test_x, test_y = test_img_list,test_mask_list


    """ Hyperparameters """
    size = (256, 256)
    #checkpoint_path = "files/PraNet-59.pth"
    checkpoint_path = "checkpoint.pth"
    checkpoint_path = "files/GMSRF-129.pth"
    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = EncoderBlock()
    print('no of params:',get_n_params(model))
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Testing """
    pred_all = []
    mask_all = []
    dice_all = []
    images_all = []
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
        images_all.append(image)
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
            #res = F.upsample(res, size=256, mode='bilinear', align_corners=False)
            #res = torch.sigmoid(res)
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
    print('the masks are saved as ',new_masks.shape)
    np.save(dataset+'gt.npy',new_masks)
    np.save(dataset+'pred.npy',pred_all)
    images_all = np.asarray(images_all)
    np.save(dataset+'images.npy',images_all)
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

test_img_list = glob("data/cvc_data/test/image/*.tif")
test_mask_list = glob("data/cvc_data/test/mask/*.tif")
dataset = 'cvc_cvc'
main_test(test_img_list,test_mask_list,dataset)
test_img_list = glob("data/kvasir_data/test/image/*.jpg")
test_mask_list = glob("data/kvasir_data/test/mask/*.jpg")
dataset = "Kvasir"
main_test(test_img_list,test_mask_list,dataset)
dataset = "CVC-ColonDB"
test_img_list = glob("data/CVC-ColonDB/images/*.tiff")
test_mask_list = glob("data/CVC-ColonDB/masks/*.tiff")
main_test(test_img_list,test_mask_list,dataset)

test_img_list = glob("data/ETIS-LaribPolypDB/ETIS-LaribPolypDB/*.tif")
test_mask_list = glob("data/ETIS-LaribPolypDB/gt/*.tif")
dataset = "ETIS-Larib"
main_test(test_img_list,test_mask_list,dataset)
