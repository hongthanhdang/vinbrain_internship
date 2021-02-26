import torchvision.transforms as transforms
from data.datasets import cifar10
from torch import nn
from models.cnn import CNN,TransferNet
from torchvision.models import resnet18
from torch.optim import SGD
from utils.metric import Accuracy

transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

dataset = {
    "class":cifar10,
    "argument":{
        "path":"/content/drive/MyDrive/datasets"
    }
}
batch_size = 1
split_train_val = 0.7
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
lr = 0.001
loss_function = nn.CrossEntropyLoss()
net = {
    "class":TransferNet,
    "net_args":{
        "model_base":resnet18,
        "pretrain":True,
        "fc_channels":[512],
        "num_classes":10
    }
}
lr_schedule = None
optimizer ={
    "class": SGD,
    "optimizer_args":{
        "momentum":0.9
    }
}
num_epochs = 10
metric = {
    "class":Accuracy,
    "metric_args":{
        "threshold": 0.5,
        "from_logits":True
    }
}
steps_save_loss = 100
output_folder = "/content/drive/MyDrive/projects/vinbrain_internship/image_classifier/train/logs"
device = "gpu"
gpu_id = 0
config_files='/content/drive/MyDrive/projects/vinbrain_internship/image_classifier/config/cifar_configs.py'
loss_file="loss_file.txt"