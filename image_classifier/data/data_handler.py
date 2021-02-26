import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import sys

sys.path.append("C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier")

from utils.utils import make_imgs
from data.men_women_dataset import MenWomenDataset

class DataHandler:
    def __init__(self, ds_class, transforms, dataset_configs, configs):
        """
        Handle with data dataloader
        Args:
            ds_class: tuple | class - 
                      (train_dataset_class, val_dataset_class) or 
                      (train_dataset_class, val_dataset_class, test_dataset_class)
                      can pass train_dataset_class ->  (train_dataset_class, train_dataset_class)
            ds_configs: tuple : dictionary:
                    root_dir
                    csv_file_path
                    transforms: class
                    label_cols_list: list
            transforms: tuple | class - (train_transforms, val_transforms) can pass train_transforms -> (train_transforms, None)
            config: img_size,bs,n_worker
        """
        self.img_size, self.bs,self.n_workers = configs['img_size'], configs['bs'],configs['n_workers']
        self.transforms = transforms
        self.ds_class=ds_class
        self.dataset_configs=dataset_configs
        self._initialize_data()
    def _initialize_data(self):
        """
        generate train, valiation, and test dataloader
        """
        torch.cuda.empty_cache()
        self.train_ds = self._create_ds(
            ds_class=self.ds_class[0], transforms=self.transforms[0], img_size=self.img_size, dataset_config=self.dataset_configs[0])
        self.val_ds = self._create_ds(
            ds_class=self.ds_class[1], transforms=self.transforms[1], img_size=self.img_size, dataset_config=self.dataset_configs[1])
        self.train_dl = self._create_dl(self.train_ds, shuffle=True)
        self.val_dl = self._create_dl(self.val_ds, shuffle=True)
        self.test_ds = None
        if len(self.ds_class) == 3:
            self.add_test_dl()
        else:
            self.test_dl = None

    def _create_ds(self, ds_class, transforms=None, img_size=None, dataset_config=None):
        return ds_class(root_dir=dataset_config['root_dir'], csv_file_path=dataset_config['csv_file_path'],img_size=dataset_config['img_size'],
                        label_cols_list=dataset_config['label_cols_list'], transforms=transforms)

    def _create_dl(self, dataset, shuffle, bs=None, **kwargs):
        return DataLoader(dataset=dataset, batch_size=bs or self.bs, shuffle=shuffle,
                          num_workers=self.n_workers, pin_memory=True)

    def add_test_dl(self, test_ds):
        self.test_ds = self._create_ds(
            test_ds, self.transforms[1], img_size=self.img_size)
        self.test_dl = self._create_dl(self.test_ds, shuffle=False)

    def show_batch(self, n_row=1, n_col=8, mode='train'):
        """
        function to show images batch that is fed for model

        Args:
            n_now: number of images in a row
            n_col:number of images in a column
            mode: `train` or `valid` to specify the img from each session
        """
        n_row = self.bs if n_row >= self.bs else n_row
        ds = {
            'train': self.train_ds,
            'valid': self.val_ds,
            'test': self.test_ds or self.val_ds
        }.get(mode, self.train_ds)

        for _ in range(n_row):
            idx = np.random.randint(len(ds), size=n_col)
            xb = torch.stack([ds[i][0] for i in idx], dim=0)
            make_imgs(xb, n_row=n_col, plot=True)


if __name__ == "__main__":
    train_dataset_config = {
        'root_dir': "C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1",
        "csv_file_path": 'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\train.csv',
        'img_size':224,
        'label_cols_list': ['Labels']
    }
    test_dataset_config = {
        'root_dir': "C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1",
        'csv_file_path': 'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\test.csv',
        'img_size':224,
        'label_cols_list': ['Labels']
    }
    cfgs = {'img_size': 224, 'bs': 2, 'n_workers': 2}

    # test_dataset = MenWomenDataset(
    #     root_dir, test_csv_file_path, label_cols_list=label_cols_list, cfg)
    # train_dataset = MenWomenDataset(root_dir, train_csv_file_path, label_cols_list=label_cols_list, cfg)
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    transforms=[transform_train,transform_test]
    dataset_configs = (train_dataset_config, test_dataset_config)
    datahandler = DataHandler(ds_class=(MenWomenDataset, MenWomenDataset), transforms=transforms, dataset_configs=dataset_configs, configs=cfgs)
    datahandler.show_batch(1,6,mode='train')
    # from tqdm import tqdm
    # pbar = tqdm(datahandler.train_dl, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)
    

    # pbar = tqdm(datahandler.val_dl, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)
