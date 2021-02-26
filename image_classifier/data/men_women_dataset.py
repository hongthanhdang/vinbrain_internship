import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose,ToPILImage,Resize,CenterCrop,ToTensor,Normalize
from skimage import io
import os
import numpy as np
from functools import partial
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
    def __init__(self, root_dir,csv_file_path, img_size,transforms = None,label_cols_list=None,configs=None):
        """
        generate menwomen dataset  
        Args:
            root_dir: number of images in a row
            csv_file_path: number of images in a column
            configs: dictionary contain configs
            label_cols_list: list of column contain label
        """
        self.img_size=img_size
        self.transforms = transforms or self.default_transforms()
        self.data_dir = root_dir

        self.df = pd.read_csv(csv_file_path)
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
            return self.transforms(img),label
        return self.transform(img)

    def _imread(self,img_path):
        image=io.imread(os.path.join(self.data_dir,img_path))
        return image[:,:,:3]

    @staticmethod
    def default_transforms():
        return lambda img_size: Compose([ToPILImage(), Resize(int(img_size*1.3)), CenterCrop((img_size, img_size)), ToTensor(), 
                                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# def split_train_test_folder(input_folder, split_range):
#     '''
#     split train - test data of dataset have no test folder
#     folder structure
#     -----
#     class1
#     class2
#     '''
#     list_class = os.listdir(input_folder)
#     list_img_name = []
#     list_label = []
#     class_number = 0
#     for i, _class in enumerate(list_class):
#         if os.path.isfile(os.path.join(input_folder, _class)):
#             continue
#         for img_name in os.listdir(os.path.join(input_folder, _class)):
#             list_img_name.append(_class + "," + img_name)
#             list_label.append(class_number)
#         class_number+=1
#     length_ = len(list_img_name)
#     list_index = np.arange(length_)
#     np.random.shuffle(list_index)
    
#     list_img_name = np.array(list_img_name)
#     list_label = np.array(list_label)

#     list_train = list_img_name[list_index[:int(length_*split_range)]]
#     list_test = list_img_name[list_index[int(length_*split_range):]]
#     list_label_train = list_label[list_index[:int(length_*split_range)]]
#     list_label_test = list_label[list_index[int(length_*split_range):]]

#     save_train_img(list_train, list_label_train, os.path.join(input_folder, 'train.txt'))
#     save_train_img(list_test, list_label_test, os.path.join(input_folder, 'test.txt'))

# def save_train_img(list_img, list_label, file_path):
#     '''
#     save list image name to file
#     inputs:
#         - list_img: List of string - list image name
#         - list_label: List of string - list label name
#         - file_path: String - file output
#     output:
#         - file path contain image and label folowing pattern: img_name,label
#     '''
#     assert(len(list_img) == len(list_label))
#     if os.path.isfile(file_path):
#         os.remove(file_path)
#     file_ = open(file_path, "w")
#     for (img, label) in zip(list_img, list_label):
#         file_.writelines(img+","+str(label)+"\n")
#     file_.close()
# if __name__ == "__main__":
    # root_dir="C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1"
    # test_csv_file_path='C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\test.csv'
    # label_cols_list=['Labels']
    # test_dataset = MenWomenDataset(root_dir, test_csv_file_path, cfg,label_cols_list=label_cols_list)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,
    #                                                batch_size=cfg['batch_size'],
    #                                                pin_memory=True)
    # from tqdm import tqdm
    # pbar = tqdm(test_dataloader, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)


    # train_csv_file_path='C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\train.csv'
    # train_dataset = MenWomenDataset(root_dir, train_csv_file_path,label_cols_list=label_cols_list,img_size, configs=cfg)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                batch_size=cfg['batch_size'],
    #                                                pin_memory=True)

    # from tqdm import tqdm
    # pbar = tqdm(train_dataloader, ncols=80, desc='Training')
    # for step, minibatch in enumerate(pbar):
    #     print(minibatch[1].shape)
        