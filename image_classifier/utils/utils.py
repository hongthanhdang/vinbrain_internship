import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import os
import cv2
import torch
from torch.nn import Module
import torchvision

class Denormalize(Module):
    """
    !! not a transform function !!
    Batch denormalization Module
    Can be use for image showing
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), device=torch.device('cpu')):
        super(Denormalize, self).__init__()
        self.mean,self.std=torch.FloatTensor(mean).to(device),torch.FloatTensor(std).to(device)
    def forward(self, x):
        return x*self.mean.view(1, 3, 1, 1) + self.std.view(1, 3, 1, 1)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_imgs(xb, n_row=8, denorm=Denormalize(), device=torch.device('cpu'), plot=True):
    xb = xb[:n_row].to(device)
    xb = denorm(xb) 
    grid_img = torchvision.utils.make_grid(xb.cpu(), nrow=n_row, normalize=True, pad_value=1)
    grid_img = grid_img.permute(1, 2, 0)
    if not plot: 
        return ToPILImage()((grid_img.numpy()*255).astype(np.uint8))
    plt.close(); plt.figure(figsize=(30,30)); plt.imshow(grid_img); plt.show()
def show_img(image):
    "show an image tensor"
    image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

def conver_numpy_image(image):
    image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    return image
def contour(images, masks):
    main = images.copy()
    _,contours,_ = cv2.findContours(masks,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i,c in enumerate(contours):
        colour = RGBforLabel.get(2)
        cv2.drawContours(main,[c],-1,colour,1)
    return main

def save_loss_to_file(file_, epoch, step, loss_train, loss_val, acc_val, lr):
    '''
    target: save loss to the file
    input:
        - file_: file contain loss
        - epoch: Interger
        - Step: Interger
        - loss_train, loss_val, acc_val: float
        - lr: float
    '''
    file_ = open(file_, "a+")
    file_.writelines("Epoch %d step %d\n"%(epoch, step))
    file_.writelines("\tLoss average %f\n"%(loss_train))
    file_.writelines("\tLoss valid average %f, acc valid %f\n"%(loss_val, acc_val))
    file_.writelines("learning_rate %f\n"%(lr))

def len_train_datatset(dataset_dict, transform, split_train_val):
    '''
    target: get train_dataset from unsplit dataset
    input:
        - dataset_dict: Dictionary contain dataset information
        - transform 
        - split_train_val: ratio split
    '''
    DatasetClass = dataset_dict["class"]
    train_dataset = DatasetClass(dataset_dict["argument"],transform = transform, mode = "train")
    return len(train_dataset)*split_train_val

def read_csv(path, normalize_columns=True, thresh=0.5, **kwargs):
    """
    
    """
    df = pd.read_csv(path, **kwasrgs)
    if normalize_columns:
        df.columns = [c.replace(' ', '_') for c in df.columns]
    if thresh is not None:
        def fix_str(x):
            if isinstance(x, str):
                if all([(c.isdigit() or c=='.') for c in x]):
                    return float(x)
                return 0.0
            elif isinstance(x, (int, float)):
                return float(x)
            return 0.0
        df[df.columns[1:]] = df[df.columns[1:]].applymap(fix_str)
        df[df.columns[1:]] = (df[df.columns[1:]].values.astype(np.uint8)  >= thresh).astype(np.uint8) 
    return df


