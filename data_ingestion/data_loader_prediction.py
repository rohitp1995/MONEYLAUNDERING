import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger
from src.utils.common import read_config
from database_operation.mongo_operation import MongodbOperations
from DataValidation.TestDataValidation import ValidateFile 
from src.utils.utils import get_unique_tablename
import pandas as pd
import datetime

class PredictionDataGetter:

    def __init__(self, filename, config_path):

        self.filename = filename
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()
        self.validator = ValidateFile(self.filename)
        self.config_path = config_path
        self.config = read_config(self.config_path)
        self.db_ops = MongodbOperations(self.config['database']['username'],  self.config['database']['pwd'])


    def validate_before_load(self):

        try:
            self.logger.info('Started validating the file')
            colsize_val = self.validator.validate_colsize()
            dtypes_val  = self.validator.validate_coldatatypes()

            if all([colsize_val, dtypes_val]):
                return True
            else:
                return False

        except Exception as e:
            self.logger.error('Issue while validating data before loading ' + str(e))
            sys.exit(1)
            

    def load_data_in_db(self):
        
        try:
        
            columns = self.validator.getcolumnlist()
            self.logger.info('DataFrame Stored for training data')
            ### Connecting database for table creation and insertion
            database = self.config['database']['db_name']
            table_name = get_unique_tablename('testing_dataset')
            self.db_ops.CreateCollection(database, table_name)
            self.db_ops.InsertManyRecord(database, self.filename.to_dict('records'), table_name)

        except Exception as e:
            self.logger.error('There was an issue while loading data in Database'+ str(e))
            sys.exit(1)

    def get_data(self):

        try:
            self.logger.info('Getting data from the file')
            columns = self.validator.getcolumnlist()
            self.data = pd.read_csv(self.filename, usecols = columns)     
            return self.data

        except Exception as e:
            self.logger.error('There was an issue fetching the data '+ str(e))
            sys.exit(1)


        





