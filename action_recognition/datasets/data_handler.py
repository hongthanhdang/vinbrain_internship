import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import sys

sys.path.append("/home/thanhdang/projects/vinbrain_internship/action_recognition/datasets")


import datasets

class DataHandler:
    def __init__(self, ds_class, transforms, dataset_configs, dataloader_configs):
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
        self.bs,self.n_workers = dataloader_configs['bs'],dataloader_configs['n_workers']
        self.transforms = (None, None)
        self.ds_class=ds_class
        self.dataset_configs=dataset_configs
        self._initialize_data()
    def _initialize_data(self):
        """
        generate train, valiation, and test dataloader
        """
        torch.cuda.empty_cache()
        self.train_ds = self._create_ds(
            ds_class=self.ds_class[0], transforms=self.transforms[0],dataset_config=self.dataset_configs[0])
        self.val_ds = self._create_ds(
            ds_class=self.ds_class[1], transforms=self.transforms[1], dataset_config=self.dataset_configs[1])
        self.train_dl = self._create_dl(self.train_ds, shuffle=True)
        self.val_dl = self._create_dl(self.val_ds, shuffle=True)
        self.test_ds = None
        if len(self.ds_class) == 3:
            self.add_test_dl()
        else:
            self.test_dl = None

    def _create_ds(self, ds_class, transforms=None, dataset_config=None):
        return ds_class(root=dataset_config['root'], source=dataset_config['source'],
                        phase=dataset_config['phase'], modality=dataset_config['modality'],
                        is_color=dataset_config['is_color'],new_length=dataset_config['new_length'],
                        new_width=dataset_config['new_width'],new_height=dataset_config['new_height'],
                        video_transform=transforms, num_segments=dataset_config['num_segments'])

    def _create_dl(self, dataset, shuffle, bs=None, **kwargs):
        return DataLoader(dataset=dataset, batch_size=bs or self.bs, shuffle=shuffle,
                          num_workers=self.n_workers, pin_memory=True)

    def add_test_dl(self, test_ds):
        self.test_ds = self._create_ds(
            test_ds, self.transforms[1], img_size=self.img_size)
        self.test_dl = self._create_dl(self.test_ds, shuffle=False)

    # def show_batch(self, n_row=1, n_col=8, mode='train'):
    #     """
    #     function to show images batch that is fed for model

    #     Args:
    #         n_now: number of images in a row
    #         n_col:number of images in a column
    #         mode: `train` or `valid` to specify the img from each session
    #     """
    #     n_row = self.bs if n_row >= self.bs else n_row
    #     ds = {
    #         'train': self.train_ds,
    #         'valid': self.val_ds,
    #         'test': self.test_ds or self.val_ds
    #     }.get(mode, self.train_ds)

    #     for _ in range(n_row):
    #         idx = np.random.randint(len(ds), size=n_col)
    #         xb = torch.stack([ds[i][0] for i in idx], dim=0)
    #         make_imgs(xb, n_row=n_col, plot=True)


if __name__ == "__main__":
    input_size = int(224 * arg.scale)
    width = int(340 * arg.scale)
    height = int(256 * arg.scale)
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    train_dataset_config = {
        'root':'/workspace/thanhdang/datasets/RWF-2000_v2/RWF-2000-raw-frames',
        'source':'/home/thanhdang/projects/vinbrain_internship/action_recognition/datasets/settings/rwf_2000/train_rgb.txt',
        'phase':"train",
        'modality':'rgb',
        'is_color':True,
        'new_length':32,
        'new_width':width,
        'new_height':height,
        'video_transform':train_transform,
        'num_segments':32
    }
    test_dataset_config = {
        'root':'/home/thanhdang/datasets/RWF-2000_v2/RWF-2000-raw-frames',
        'source':'/home/thanhdang/projects/vinbrain_internship/action_recognition/datasets/settings/rwf_2000/train_rgb.txt',
        'phase':"val",
        'modality':'rgb',
        'is_color':True,
        'new_length':32,
        'new_width':width,
        'new_height':height,
        'video_transform':val_transform,
        'num_segments':32
    }
    dl_cfgs = {'bs': 2, 'n_workers': 2}

    

    transforms=[transform_train,transform_test]

    dataset_configs = (train_dataset_config, test_dataset_config)
    datahandler = DataHandler(ds_class=(MenWomenDataset, MenWomenDataset), transforms=None, dataset_configs=dataset_configs, dataloader_configs=dl_cfgs)


    from tqdm import tqdm
    pbar = tqdm(datahandler.train_dl, ncols=80, desc='Training')
    for step, minibatch in enumerate(pbar):
        print(minibatch[1].shape)
    

    # pbar = tqdm(datahandler.val_dl, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)