from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from logs.logger import logging
import torch


class Data_transform():

    def __init__(self, data: str):
        self.data = data

    def data_folder(self):
        
        try:
            Data_folder = datasets.ImageFolder(root=self.data,
                                 transform=self._transformation(),
                                 target_transform=None)
            logging.info("Data folder created")
            return Data_folder
        
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e
            
    def _transformation(self):
        trans =v2.Compose([
            v2.Resize((224,224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        return trans
    
class Split_data(Data_transform): 
    
    def __init__(self, data):
        super().__init__(data)

    def data_loader(self):
        try:
            img_train, img_test = self._train_test_split()
            
            train_dataloader=DataLoader(dataset=img_train,
                                        batch_size=16,
                                        shuffle=True)

            test_dataloader=DataLoader(dataset=img_test,
                                        batch_size=16,
                                        shuffle=False)
            
            logging.info("Train and test dataloaders created")
            return train_dataloader, test_dataloader
        except Exception as e:
            logging.error("Error in preprocessing data into dataloaders {}".format(e))
            raise e
    
    def _train_test_split(self):
        try:
            img_train, img_test = train_test_split(self.data_folder(), test_size=0.2,random_state=42)
            logging.info("Data splited to train and test set")
            return img_train, img_test
        
        except Exception as e:
            logging.error("Error in preprocessing data into train and test set {}".format(e))
            raise e
        
    def targets(self):
        try:    
            img_train_target, img_test_target = train_test_split(self.data_folder().targets, test_size=0.2,random_state=42)
            logging.info("Targets splited to train and test set")
            return img_train_target, img_test_target
        
        except Exception as e:
            logging.error("Error in preprocessing data into targets {}".format(e))
            raise e






        

