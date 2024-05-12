from pipeline import pipeline
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
data= os.path.join(os.getcwd(), "Data", "Jellyfish")

if __name__ == '__main__':
    pipeline(data, device).TinyVGG_flow()
    pipeline(data, device).Transfer_learning_flow()