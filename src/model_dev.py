from torch import nn
from torchvision import models
from logs.logger import logging

class TinyVGG_Jellyfish(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.CNNblock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.CNNblock2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.outputlayer=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56,
                      out_features=output_shape)
            )
    def forward(self, x):
        model = self.outputlayer(self.CNNblock2(self.CNNblock1(x)))
        return model
    
class Transfer_learning():

    def model(in_features: int, out_features: int, bias: bool):
        try:    
            model = models.efficientnet_v2_s(weights='DEFAULT')
            model.classifier = nn.Linear(in_features= in_features, out_features= out_features, bias=bias) 
            logging.info("Model succesfully imported")
            return model
        except Exception as e:
            logging.error("Error in preprocessing data {}".format(e))
            raise e  
            
    
