import matplotlib.pyplot as plt
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from logs.logger import logging

class Evaluation():

    def __init__(self, model, path, device, dataloader, targets):
        self.model = model
        self.path = path
        self.device = device
        self.dataloader = dataloader
        self.targets = targets

    def plots(self, train_losses, test_losses, accuracies, epoch_list):
        try:
            plt.subplot(2,1,1)
            plt.title('Train-Test Loss')
            plt.plot(epoch_list, torch.tensor(train_losses), c="green", label='Train')
            plt.plot(epoch_list, torch.tensor(test_losses), c="blue", label='Test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.tight_layout(w_pad=5, pad=2)

            plt.subplot(2,1,2)
            plt.title('Accuracy score')
            plt.plot(epoch_list, torch.tensor(accuracies), c="red")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.tight_layout(w_pad=5, pad=2)

            plt.show()
            logging.info("Plot created")

        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e


    def confusion_matrix(self, classes):
        try:
            matrix=ConfusionMatrix(num_classes=len(classes), task='multiclass')
            matrix_tensor=matrix(preds=torch.cat(self._load_predictions()), target=torch.tensor(self.targets))

            plot_confusion_matrix(
                conf_mat=matrix_tensor.numpy(),
                class_names=classes,
                figsize=(5,5))
            plt.title('Confusion Matrix')
            plt.show()
            logging.info("Confusion matrix created")
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e        

    def _load_predictions(self): 
        try:    
            self.model.load_state_dict(torch.load(self.path))
            y_predicted=[]
            self.model.eval()
            with torch.inference_mode():
                for A, b in self.dataloader:
                    A,b= A.to(self.device), b.to(self.device)

                    test_pred = self.model(A)
                    y_predicted.append((torch.softmax(test_pred, dim=1).argmax(dim=1)).cpu())

            logging.info("Predictions created")
            return y_predicted
        
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e
   
    




