from pipeline import pipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

data="C:\\Users\\Marcel\\Desktop\\Portfolio\\ComputerVision project\\Data\\Jellyfish"
path="C:\\Users\\Marcel\\Desktop\\Portfolio\\ComputerVision project\\Trained_model\\TinyVGG_best"

pipeline(data, path, device).TinyVGG_flow()