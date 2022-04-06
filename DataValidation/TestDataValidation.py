import pandas as pd
import sys
import os
import csv
import json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger


f = open('DataValidation/Prediction_schema.json')
Predictionvalidation = json.load(f)

class ValidateFile:
    
    def __init__(self, file_data):
        self.df = file_data
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def validate_colsize(self):

        try:
            self.logger.info('Started validating size of the columns in the file')
            if self.df.columns.nunique() == Predictionvalidation['NumberofColumns']:
                return True
            else:
                return False
              
        except Exception as e:
            self.logger.error('There was error while validating column size: '+ str(e))
            sys.exit(1)

    def validate_coldatatypes(self):

        try:
            self.logger.info('started validating datatypes for the columns in the file')
            if  self.validate_colsize():
                col_dict = {}
                for col in self.df.columns:
                    if col in Predictionvalidation['ColName']:
                        if (self.df[col].dtype) == Predictionvalidation['ColName'][col]:
                            col_dict['Datatype_'+col] = 'Valid'
                        else:
                            col_dict['Datatype_'+col] = 'InValid'
            
            for key,value in col_dict.items():
                if value == 'InValid':
                    self.logger.info(f'{key} has a wrong datatype')
                
            if 'invalid' not in col_dict.values():
                return True
            else:
                return False

        except Exception as e:
            self.logger.error('There is error while validating column datatypes: '+str(e))
            sys.exit(1)
        

    def getcolumnlist(self):

        try:
            self.logger.info('getting column names to store in dataframe')
            col_list = list(Predictionvalidation['ColName'].keys())
            return col_list

        except Exception as e:
            self.logger.error('There is error while getting column list: ' + str(e))
            sys.exit(1)



    
            

            


                

