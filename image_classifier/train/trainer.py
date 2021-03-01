import torch
import os
# from shutil import copy
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import SGD
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

import sys
sys.path.append("C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier")
from utils.utils import save_loss_to_file,AverageMeter
from models.cnn import CNN, TransferNet
from data.datasets import ListDataset
from utils.metric import Accuracy
from data.data_handler import DataHandler
from data.men_women_dataset import MenWomenDataset


class Trainer:
    def __init__(self,net, data, optimizer, crition,transform_test, metric,lr_scheduler=None,  configs=None):
        '''
        target: initialize trainer for training
        inputs:
            - configs contains parameter for training: 
                - lr, batch_size, num_epoch, steps_save_loss, output_folder, device
                - loss_function
                - net: dict - contrain information of model
                - optimizer: dict - contain information of optimizer
                - transform: use for predict list image
                - lr_schedule: dict - contain information for schedule learning rate
                - metric: dict - information of metric for valid and test
                - loss_file: String - name of file in output_folder contain loss training process
            - data: instance Data classes in data folder
        '''
        self.lr = configs['lr']
        self.batch_size = data.bs
        self.num_epochs = configs['num_epochs']
        self.crition = crition
        self.net = net
        self.optimizer = optimizer
        self.transform_test = transform_test
        self.n_crops=configs['n_crops']
        # data
        self.data = data
        # evaluate
        self.metric = metric

        # schedule learning rate
        if lr_scheduler is not None:
            steps_per_epoch=len(data.train_dl)
            self.lr_scheduler = lr_scheduler(self.optimizer,max_lr=0.0005,epochs=self.num_epochs,steps_per_epoch=steps_per_epoch)
        # else:
        #     self.lr_scheduler = lr_scheduler
        #     self.lr_shedule_metric = lr_schedule["metric"]
        #     self.lr_schedule_step_type = lr_schedule["step_type"]

        # training process
        self.current_epoch = 0
        self.list_loss = []
        self.steps_save_loss = configs['steps_save_loss']
        self.output_folder = configs['output_folder']
        self.config_files = configs['config_files']

        # define loss file
        self.loss_file = configs['loss_file']

        # config cuda
        cuda = configs['device']
        self.device = torch.device(
            cuda if cuda == "cpu" else "cuda:"+str(configs['gpu_id']))
        self.net.to(self.device)

        # config output
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        # copy(self.config_files, self.output_folder)

        # tensorboard
        self.summaryWriter = SummaryWriter(self.output_folder)
        self.global_step = 0

    def train(self, loss_file=None):
        '''
        target: training the model
        input:
            - loss_file: file contain loss of training process
        '''
        if loss_file is not None:
            self.loss_file = loss_file
        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

            # if self.lr_scheduler is not None:
            #     if self.lr_schedule_step_type == "epoch":
            #         self.schedule_lr()

    def test(self):
        '''
        target: test the model
        '''
        loss, acc = self.evaluate("test")
        print("Test loss: %f test acc %f" % (loss, acc))

    def train_one_epoch(self):
        '''
        target: train per epoch
            - load image form train dataloader
            - train
            - save train result to summary writer
            - update learning rate if necessary
        '''

        train_loss = 0
        for i, sample in enumerate(self.data.train_dl): 
            self.net.train()
            xb, yb = sample[0].to(
                self.device), sample[1].to(self.device)
            bs, n_crops, c, h, w = xb.size()
            xb = xb.view(-1, c, h, w)
            outputs = self.net(xb)
            if self.n_crops > 0:
                outputs = outputs.view(bs, n_crops, -1).mean(1)
            loss = self.crition(outputs, yb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.lr_scheduler.step()
            # if self.lr_scheduler is not None:
            #     if self.lr_schedule_step_type == "batch":
            #         self.schedule_lr(i)
            self.summaryWriter.add_scalar(
                'learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.summaryWriter.add_scalars('loss',
                                           {
                                               'loss_train': loss.item()
                                           }, self.global_step)
            if i % (self.steps_save_loss-1) == 0:
                print("Epoch %d step %d" % (self.current_epoch, i))
                train_loss_avg = train_loss/self.steps_save_loss
                print("\tLoss average %f" % (train_loss_avg))
                val_loss_avg, val_acc_avg = self.evaluate(mode="val")
                print("\tLoss valid average %f, acc valid %f" %
                      (val_loss_avg, val_acc_avg))
                print("learning_rate ", self.optimizer.param_groups[0]['lr'])
                train_loss = 0.0
                loss_file_path = os.path.join(
                    self.output_folder, self.loss_file)
                save_loss_to_file(loss_file_path, self.current_epoch, i, train_loss_avg,
                                  val_loss_avg, val_acc_avg, self.optimizer.param_groups[0]['lr'])
                self.summaryWriter.add_scalars(
                    'loss', {'loss_val': val_loss_avg}, self.global_step)
                self.summaryWriter.add_scalars(
                    'acc', {'acc_val': val_acc_avg}, self.global_step)
            self.global_step += 1

    # def schedule_lr(self, iteration=0):
    #     '''
    #     target: update learning rate schedule
    #     input:
    #         -iteration: Interger - iteration of each epoch, using for mode batch
    #     '''
    #     if not self.lr_scheduler is None:
    #         if self.lr_shedule_metric is not None:
    #             if self.lr_shedule_metric == "epoch":
    #                 self.lr_scheduler.step(
    #                     self.current_epoch+iteration/self.batch_size)
    #             else:
    #                 val_loss, val_acc = self.evaluate(mode="val")
    #                 self.lr_scheduler.step(eval(self.lr_shedule_metric))
    #         else:
    #             self.lr_scheduler.step()

    def evaluate(self, mode="val", metric=None):
        '''
        target: caculate model with given metric
        input:
            - mode: String - ["train", "val", "test"]
            - metric: class of metric
        output:
            - loss: average loss of whole dataset
            - metric_value
        '''
        self.net.eval()
        val_loss_value= AverageMeter()
        if metric is None:
            metric = self.metric
        loader = {
            "val": self.data.val_dl,
            "train": self.data.train_dl,
            "test": self.data.test_dl
        }
        preds,targets=[],[]
        with torch.no_grad():
            for xb,yb in loader[mode]:
                xb,yb = xb.to(self.device),yb.to(self.device)
                output = self.net(xb)
                loss = self.crition(output, yb)
                preds.append(output)
                targets.append(yb)
                val_loss_value.update(val=loss.item())
            # import pdb; pdb.set_trace()
            preds_tensor = torch.cat(preds)
            targets_tensor = torch.cat(targets)
            metric_result = metric(preds_tensor, targets_tensor)
            return val_loss_value.avg, metric_result

    def get_prediction(self, list_img):
        '''
        targets: get output of model from given list of images
        inputs:
            - list_img: list of image
        output: list of outputs model
        '''
        dataset = ListDataset(list_img, self.transform_test)
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=self.batch_size)
        self.net.eval()
        output_list = []
        with torch.no_grad():
            for i, images in enumerate(dataloader):
                images = images.to(self.device)
                outputs = self.net(images)
                output_list.append(outputs)
        return torch.cat(output_list)

    def predict(self, img):
        '''
        targets: get output of model from given 1 image
        inputs:
            - img: Image from image io
        output: list of outputs model
        '''
        self.net.eval()
        with torch.no_grad():
            img_tensor = self.transform_test(img)
            img_tensor = img_tensor.to(self.device)
            output = self.net(img_tensor)
            return output

    def num_correct(self, outputs, labels):
        '''
        target: calculate number of element true
        '''
        _, predicted = torch.max(outputs, 1)
        return (predicted == labels).sum().item()

    def save_checkpoint(self, filename=None):
        '''
        target: save checkpoint to file
        input:
            - filename: String - file name to save checkpoint
        '''
        if filename is None:
            filename = "checkpoint_%d" % (self.current_epoch)
        file_path = os.path.join(self.output_folder, filename)
        torch.save(self.net.state_dict(), file_path)

    def load_checkpoint(self, filename=None):
        '''
        target: load checkpoint from file
        input:
            - filename: String - file name to load checkpoint
        '''
        if filename is None:
            filename = "checkpoint_%d" % (self.num_epochs-1)
        file_path = os.path.join(self.output_folder, filename)
        self.net.load_state_dict(torch.load(
            file_path, map_location=self.device))


if __name__ == "__main__":
    # dataset building
    train_dataset_config = {
        'root_dir': "C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1",
        "csv_file_path": 'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\train.csv',
        'img_size':224,
        'label_cols_list': ['Labels'],
        'imagenet':True,
        'img_size':256,
        'crop_size':250,
        'n_crops':5,
        'pixel_mean':128,
        'pixel_std':50
    }
    test_dataset_config = {
        'root_dir': "C:\\Users\\thanhdh6\\Documents\\datasets\\menwomen1",
        'csv_file_path': 'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\data\\test.csv',
        'img_size':224,
        'label_cols_list': ['Labels'],
        'imagenet':True,
        'mode':'val',
        'crop_size':250,
    }
    cfgs = {'img_size': 224, 'bs': 2, 'n_workers': 2}

    # test_dataset = MenWomenDataset(
    #     root_dir, test_csv_file_path, label_cols_list=label_cols_list, cfg)
    # train_dataset = MenWomenDataset(root_dir, train_csv_file_path, label_cols_list=label_cols_list, cfg)
    # transform_train = transforms.Compose([
    #     transforms.ToPILImage(),
    #     # transforms.ToTensor(),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    # ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    # transforms=[transform_train,transform_test]

    dataset_configs = (train_dataset_config, test_dataset_config)
    datahandler = DataHandler(ds_class=(MenWomenDataset, MenWomenDataset), transforms=None, dataset_configs=dataset_configs, configs=cfgs)
    # datahandler.show_batch(1,6,mode='train')

    # trainer buiding
    trainer_configs = {
        'model_path': '',
        'validate': 0.7,
        'lr': 0.001,
        'num_epochs': 10,
        'steps_save_loss': 2,
        'output_folder': 'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\train\\logs',
        'device': 'cpu',
        'gpu_id': 0,
        'lr_schedule':None,
        'config_files':'C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\configs\\cifar_configs.py',
        'loss_file':"loss_file.txt",
        'n_crops':5

    }
    
    net = TransferNet(model_base=resnet18, pretrain=True,
                      fc_channels=[2048], num_classes=2)
    optimizer_config={
        'momentum':0.9
    }
    optimizer = SGD(net.parameters(),lr=1e-4,momentum=0.9)
    metric = Accuracy(threshold=0.5, from_logits=True)
    crition= nn.CrossEntropyLoss()
    trainer = Trainer(net, datahandler, optimizer, crition,transform_test, metric, lr_scheduler=OneCycleLR,configs=trainer_configs)
    trainer.train()
    # trainer.load_checkpoint("C:\\Users\\thanhdh6\\Documents\\projects\\vinbrain_internship\\image_classifier\\train\\logs\\checkpoint_9")
    # print(trainer.evaluate(mode="val",metric=Accuracy()))

