import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from logs.logger import logging

class CustomModelTrainer:
    def __init__(self, model, train_dataloader, save_path, test_dataloader, device='cuda'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.save_path = save_path
        

    def train_loop(self, epochs, lr):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        train_losses = []
        test_losses = []
        accuracies = []
        epoch_list = []
        best_epoch = []

        start_time = timer()

        try:
            for epoch in tqdm(range(epochs)):
                logging.info(f"Epoch: {epoch}\n-------")

                train_loss = self._training(optimizer, loss_fn)
                train_loss /= len(self.train_dataloader)
                train_losses.append(train_loss)

                test_loss, test_acc = self._evaluate(loss_fn)
                test_losses.append(test_loss)
                accuracies.append(test_acc)
                epoch_list.append(epoch)

                logging.info(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

                if test_acc == max(accuracies) and test_loss == min(test_losses):
                    self._save_model()
                    best_epoch.append(epoch)

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        end_time = timer()
        logging.info(f"""
        [INFO] Total training time: {end_time-start_time:.3f} seconds
               The Best predictions were made in epoch number {max(best_epoch)}:
                  - Test loss: {test_losses[max(best_epoch)]:.5f}
                  - Accuracy: {accuracies[max(best_epoch)]*100:.2f}%""")

        return train_losses, test_losses, accuracies, epoch_list

    def _evaluate(self, loss_fn):
        test_loss, test_acc = 0, 0
        self.model.eval()

        with torch.inference_mode():
            for A, b in self.test_dataloader:
                A, b = A.to(self.device), b.to(self.device)
                test_pred = self.model(A)
                test_loss += loss_fn(test_pred, b)
                test_acc += accuracy_score(b.cpu(), (torch.softmax(test_pred, dim=1).argmax(dim=1)).cpu())

        test_loss /= len(self.test_dataloader)
        test_acc /= len(self.test_dataloader)

        return test_loss, test_acc
    
    def _training(self, optimizer, loss_fn):
        
        train_loss = 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            self.model.train()
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            loss.backward()
            optimizer.step()

            if batch % 16 == 0:
                logging.info(f"Samples {batch * len(X)}/{len(self.train_dataloader.dataset)}")

        return train_loss

    def _save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        logging.info("Model's parameters saved\n")       
