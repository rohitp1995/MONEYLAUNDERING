import pandas as pd
import sys
import os
import csv
import json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger


f = open('DataValidation/Training_schema.json')
trainingvalidation = json.load(f)

class ValidateFile:
    
    def __init__(self, file):
        self.file = file
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()


    def validate_extension(self):
        
        try:
            if '.csv' in self.file:
                return True
            else:
                return False
        except Exception as e:
            self.logger.error('There was some error while validating file extension: '+ str(e))
            sys.exit(1)
            

    def validate_colsize(self):
        
        try:
            if self.validate_extension():
                self.df = pd.read_csv(self.file)
                if self.df.columns.nunique() == trainingvalidation['NumberofColumns']:
                    return True
                else:
                    return False
        except Exception as e:
            self.logger.error('There was error while validating column size: '+ str(e))
            sys.exit(1)

    def validate_coldatatypes(self):

        try:
            if  self.validate_colsize():
                col_dict = {}
                for col in self.df.columns:
                    if col in trainingvalidation['ColName']:
                        if (self.df[col].dtype) == trainingvalidation['ColName'][col]:
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
            self.logger.error('There is error while validation column datatypes: '+ str(e))
            sys.exit(1)

    
    def getcolumnlist(self):

        try:
            self.logger.info('getting column names to store in dataframe')
            col_list = list(trainingvalidation['ColName'].keys())
            return col_list

        except Exception as e:
            self.logger.error('There is error while getting column list: ' + str(e))
            sys.exit(1)



    

            


                

