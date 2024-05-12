import os
from datetime import datetime
from logs.logger import logging

class utils():
    def __init__(self, model):
        self.model = model
        

    def create_dir(self):    
        try:
            Model_file=f"{datetime.now().strftime('_%m_%d_%Y_%H_%M_%S')}"
            model_path=os.path.join(os.getcwd(), "Trained_model", self.model) + Model_file
            os.makedirs(model_path,exist_ok=True)
            logging.info("Model directory created")
            return model_path
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
             
    
    def save_path(self, path):
        try:
            save_path = os.path.join(path, self.model)
            logging.info("Model save path created")
            return save_path
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e