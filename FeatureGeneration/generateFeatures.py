import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger
import math
import numpy as np

class GenerateNewFeatures:
        
    def __init__(self, Data):
        self.data = Data
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def make_date_columns(self, column):

        try:
            self.logger.info('Date Features Creation Started')
            self.data['day'] = [math.floor(float(x)/24) for x in self.data[column]]
            self.data['week'] = [math.floor(float(x)/168) for x in self.data[column]]
            self.data['hour'] = self.data[column]

        except Exception as e:
            self.logger.error('Feature Creation Unsucessful: '+ str(e))
            sys.exit(1)

    
    def make_diff_columns(self, MONEY_IN, MONEY_OUT):

        try:

            self.logger.info('Diff Features Creation Started')
            self.data["origdiffcheck"] = np.where((self.data['type'].isin(MONEY_OUT)) &  (self.data["oldbalanceOrg"] - self.data["newbalanceOrig"] == self.data['amount']),1,0)
            self.data["origdiffcheck"] = np.where((self.data['type'].isin(MONEY_IN)) & (self.data["oldbalanceOrg"] + self.data["newbalanceOrig"] == self.data['amount']), 1,self.data["origdiffcheck"])

            self.data["destdiffcheck"] = np.where((self.data['type'].isin(MONEY_OUT)) &  (self.data["oldbalanceDest"] + self.data["newbalanceDest"] == self.data['amount']),1,0)
            self.data["destdiffcheck"] = np.where((self.data['type'].isin(MONEY_IN)) & (self.data["newbalanceDest"] - self.data["newbalanceDest"] == self.data['amount']), 1,self.data["destdiffcheck"])

        except Exception as e:
            self.logger.error('Diff Feature Creation Unsucessful: '+ str(e))
            sys.exit(1)
            