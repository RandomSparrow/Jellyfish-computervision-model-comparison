from data_preparation import Split_data
from model_dev import Transfer_learning, TinyVGG_Jellyfish
from training_loop import CustomModelTrainer
from evaluation import Evaluation
from utils import utils

class pipeline():
   
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def Transfer_learning_flow(self):
        path = utils('Transfer_learning').create_dir()
        save_path = utils('Transfer_learning').save_path(path)
        train_dataloader, test_dataloader = Split_data(self.data).data_loader()
        model = Transfer_learning().model(1280, 6, True).to(self.device)
        train_losses, test_losses, accuracies, epoch_list = CustomModelTrainer(model, train_dataloader, save_path, test_dataloader, self.device).train_loop(10, 0.0001)
        img_train_target, img_test_target = Split_data(self.data).targets()
        Data_folder = Split_data(self.data).data_folder()
        Evaluation(model, path, self.device, test_dataloader, img_test_target).plots(train_losses, test_losses, accuracies, epoch_list)
        Evaluation(model, path, self.device, test_dataloader, img_test_target).confusion_matrix(Data_folder.classes, save_path)

    def TinyVGG_flow(self):
        path = utils('TinyVGG').create_dir()
        save_path = utils('TinyVGG').save_path(path)
        train_dataloader, test_dataloader = Split_data(self.data).data_loader()
        model = TinyVGG_Jellyfish(3,32,6).to(self.device)
        train_losses, test_losses, accuracies, epoch_list = CustomModelTrainer(model, train_dataloader, save_path, test_dataloader, self.device).train_loop(10, 0.0001)
        img_train_target, img_test_target = Split_data(self.data).targets()
        Data_folder = Split_data(self.data).data_folder()
        Evaluation(model, path, self.device, test_dataloader, img_test_target).plots(train_losses, test_losses, accuracies, epoch_list)
        Evaluation(model, path, self.device, test_dataloader, img_test_target).confusion_matrix(Data_folder.classes, save_path)    





