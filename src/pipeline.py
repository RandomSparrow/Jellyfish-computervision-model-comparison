from data_preparation import Split_data
from model_dev import Transfer_learning, TinyVGG_Jellyfish
from training_loop import CustomModelTrainer
from evaluation import Evaluation

class pipeline():
   
    def __init__(self, data, path, device):
        self.data = data
        self.path = path
        self.device = device

    def Transfer_learning_flow(self):
        train_dataloader, test_dataloader = Split_data(self.data).data_loader()
        model = Transfer_learning().model(1280, 6, True).to(self.device)
        train_losses, test_losses, accuracies, epoch_list = CustomModelTrainer(model, train_dataloader, self.path, test_dataloader, self.device).train_loop(10, 0.0001)
        img_train_target, img_test_target = Split_data(self.data).targets()
        Data_folder = Split_data(self.data).data_folder()
        Evaluation(model, self.path, self.device, test_dataloader, img_test_target).plots(train_losses, test_losses, accuracies, epoch_list)
        Evaluation(model, self.path, self.device, test_dataloader, img_test_target).confusion_matrix(Data_folder.classes)

    def TinyVGG_flow(self):
        train_dataloader, test_dataloader = Split_data(self.data).data_loader()
        model = TinyVGG_Jellyfish(3,32,6).to(self.device)
        train_losses, test_losses, accuracies, epoch_list = CustomModelTrainer(model, train_dataloader, self.path, test_dataloader, self.device).train_loop(10, 0.0001)
        img_train_target, img_test_target = Split_data(self.data).targets()
        Data_folder = Split_data(self.data).data_folder()
        Evaluation(model, self.path, self.device, test_dataloader, img_test_target).plots(train_losses, test_losses, accuracies, epoch_list)
        Evaluation(model, self.path, self.device, test_dataloader, img_test_target).confusion_matrix(Data_folder.classes)    





