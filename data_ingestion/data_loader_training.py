import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger
import pandas as pd

class TrainingDataGetter:

    def __init__(self, filename):

        self.filename = filename
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def get_data(self):
        
        try:
            self.data = pd.read_csv(self.filename)
            self.logger.info('DataFrame Stored for training data')
            return self.data
        except Exception as e:
            self.logger.error('There was an issue while creating the DataFrame'+ str(e))

        






