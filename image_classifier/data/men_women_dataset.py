import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
from functools import partial
from collections import namedtuple
from PIL import Image
import pandas as pd
import cv2
'''
config example
dataset = {
"class":MenWomenDataset,
"argument":{
    "data_path":"..//menwomen"
    "csv_file_path":"..//data//.csv
}
}

'''
class MenWomenDataset(Dataset):
    def __init__(self, root_dir,csv_file_path,mode='train',transforms = None,label_cols_list=None,configs=None):
        """
        generate menwomen dataset  
        Args:
            root_dir: number of images in a row
            csv_file_path: number of images in a column
            configs: dictionary contain configs
            label_cols_list: list of column contain label
        """
        self.transforms = transforms
        self.data_dir = root_dir
        self.configs = configs
        self.df = pd.read_csv(csv_file_path)
        self.mode=mode
        # self.transforms = partial(self.transforms, img_size=self.img_size)()
        # labels
        self.label_cols_list=label_cols_list
        if self.label_cols_list is not None:
            self.labels=self.df[self.label_cols_list].to_numpy()
    def __len__(self):
        '''
        target: length of dataset
        '''
        return len(self.df)

    def __getitem__(self, idx):
        img = self._imread(self.df['Paths'][idx])
        if self.label_cols_list is not None:
            label=self.labels[idx][0]
            # label=label.astype(int)
            # label=[label,1-label]
            # label=label.astype(int)
            if self.transforms==None:
                img_aug=matching_templates(img,self.configs,self.mode)
            else:
                img_aug=self.transforms(img)
            return img_aug,label
        return self.transform(img)

    def _imread(self,img_path):
        image=io.imread(os.path.join(self.data_dir,img_path))
        return image[:,:,:3]

    # @staticmethod
    # def default_transforms():
    #     return lambda img_size: Compose([ToPILImage(), Resize(int(img_size*1.3)), CenterCrop((img_size, img_size)), ToTensor(), 
    #                                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def matching_templates(org_img, cfg, mode='train'):
    """

    Args:
        org_img:
        cfg:
        mode:

    Returns:

    """
    # Convert cfg from dictionary to a class object
    cfg = namedtuple('cfg', cfg.keys())(*cfg.values())

    img = Image.fromarray(org_img)

    if cfg.imagenet:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(
            [cfg.pixel_mean / 256, cfg.pixel_mean / 256, cfg.pixel_mean / 256],
            [cfg.pixel_std / 256, cfg.pixel_std / 256, cfg.pixel_std / 256])

    if mode == 'train':
        if cfg.n_crops == 10:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.TenCrop(size=cfg.crop_size),
                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        elif cfg.n_crops == 5:
            trans = transforms.Compose(
                [transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.FiveCrop(size=cfg.crop_size),
                 transforms.Lambda(
                     lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                 transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
        elif cfg.n_crops == -1:
            trans = transforms.Compose([transforms.RandomResizedCrop(size=cfg.img_size, scale=(0.8, 1.0)),
                                        transforms.ToTensor(),normalize])
        else:
            trans = transforms.Compose(
                [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
                 transforms.RandomRotation(degrees=5), transforms.ColorJitter(),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    elif mode == 'test' and cfg.n_crops > 0:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.FiveCrop(size=cfg.crop_size),
             transforms.Lambda(
                 lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
             transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
    else:
        trans = transforms.Compose(
            [transforms.Resize(size=cfg.img_size), transforms.CenterCrop(size=cfg.crop_size),
             transforms.ToTensor(), normalize])

    # if mode == 'train' and cfg.n_crops == 0 and cfg.augmix:
    #     if cfg.no_jsd:
    #         return augment_and_mix(img, trans, cfg)
    #     else:
    #         return trans(img), augment_and_mix(img, trans, cfg), augment_and_mix(img, trans, cfg)
    # else:
    #     if mode == 'train' and cfg.n_crops == -1:
    #         aug = strong_aug(cfg.crop_size, p=1)
    #         img = Image.fromarray(augment(aug, org_img))

    return trans(img)

if __name__ == "__main__":
    root_dir="C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1"
    test_csv_file_path='C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\test.csv'
    label_cols_list=['Labels']
    cfg={
        'batch_size':3,
        'imagenet':False,
        'img_size':256,
        'crop_size':250,
        'n_crops':5,
        'pixel_mean':128,
        'pixel_std':50
    }
    test_dataset = MenWomenDataset(root_dir, test_csv_file_path, mode='train',label_cols_list=label_cols_list,configs=cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=cfg['batch_size'],
                                                   pin_memory=True)
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, ncols=80, desc='Training')
    for step, minibatch in enumerate(pbar):
        print(minibatch[0].shape)


    # train_csv_file_path='C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\train.csv'
    # train_dataset = MenWomenDataset(root_dir, train_csv_file_path,label_cols_list=label_cols_list,img_size, configs=cfg)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                batch_size=cfg['batch_size'],
    #                                                pin_memory=True)

    # from tqdm import tqdm
    # pbar = tqdm(test_dataloader, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)
        