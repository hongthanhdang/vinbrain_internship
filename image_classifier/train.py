from config import cifar_configs
from train.trainer import Trainer
from data.data_handler import DataHandler

if __name__ =="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config-file", default = "./config/train_config.yaml", metavar = "FILE", type = str)
    # args = parser.parse_args()
    
    # #extract config
    # config_file = open(args.config_file, 'r')
    # configs = yaml.load(config_file)
    datahandler=DataHandler(cifar_configs)
    trainer=Trainer(cifar_configs,datahandler)
    trainer.train()


