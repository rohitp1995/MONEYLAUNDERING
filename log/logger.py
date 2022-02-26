import logging
import os 

class Logger:
    def __init__(self,log_dir):
        self.log_dir = log_dir

    def logging(self):
        
        logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(filename= os.path.join(self.log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

        return logging
