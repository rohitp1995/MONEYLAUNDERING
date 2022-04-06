import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from log.logger import Logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE    
from sklearn.model_selection import train_test_split

class Preprocess:
        
    def __init__(self, Data):
        self.data = Data
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()


    def drop_columns(self, columns):

        try:
            self.logger.info('Started Column Dropping')
            self.data.drop(columns, axis = 1, inplace = True)
            self.logger.info('Succesfully dropped columns')

        except Exception as e:
            self.logger.error('Error while dropping columns: '+ str(e))
            sys.exit(1)


    def encode_categorical_columns(self):
       
        try:
           self.logger.info('Started encoding features')
    
           self.data['C_to_C'] = np.where((self.data['nameOrig'].str.startswith('C')) & (self.data['nameDest'].str.startswith('C')),1,0)
           self.data['C_to_M'] = np.where((self.data['nameOrig'].str.startswith('C')) & (self.data['nameDest'].str.startswith('M')),1,0)
           self.data['M_to_C'] = np.where((self.data['nameOrig'].str.startswith('M')) & (self.data['nameDest'].str.startswith('C')),1,0)
           self.data['M_to_M'] = np.where((self.data['nameOrig'].str.startswith('M')) & (self.data['nameDest'].str.startswith('M')),1,0)

           self.data = pd.get_dummies(self.data, prefix = 'type', columns=['type'])
           return self.data

           self.logger.info('End encoding features')
        except Exception as e:
            self.logger.error('Encoding Features was unsuccessful: '+ str(e))
            sys.exit(1)


    def Nullcheck(self):

        data_cnt = self.data.shape[0]
        cols_with_missing_values=[]
        
        try:
            self.logger.info('Checking for null values')
            null_dict = self.data.isnull().sum().to_dict()
            for col, val in null_dict.items():
                if val > 0:
                        cols_with_missing_values.append([col,(val/self.data_cnt)*100])
                        null_col_df = pd.DataFrame(cols_with_missing_values, columns=['columns','percentage'])
                        null_col_df.to_excel('data_preprocessing/Null_value_report.xlsx', index = False)
                        logger.info('Null values report generated')

        except Exception as e:
            self.logger.error('Error while checking and storing Null Values: '+ str(e))
            sys.exit(1)


    def seperate_label_columns(self, cluster_data, label_col_name):

        try:
            self.logger.info('Started Seperation of independent and dependent columns')
            X_data = cluster_data.drop(label_col_name, axis = 1)
            y_data = cluster_data[label_col_name]
            self.logger.info('Seperation of independent columns and dependent columns completed')

            return X_data, y_data
        
        except Exception as e:
            self.logger.error('Error while seperating label values: '+ str(e))
            sys.exit(1)

    def split_data(self, test_size, target_col):

        try:
            
            self.logger.info('Starting train test split of data')
            X = self.data.drop(target_col, 1)
            y = self.data[target_col]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state = 42, stratify = y)

            return X_train, X_val, y_train, y_val      

        except Exception as e:
            self.logger.error('Error while performing train test split: '+ str(e))
            sys.exit(1)


    def scalenumericvalues(self, num_columns):

        try:
            self.logger.info('Started Scaling of Numeric Columns')
            sc = StandardScaler()
            scaled_data = pd.DataFrame(sc.fit_transform(self.data[num_columns]), columns=num_columns)
            self.data.drop(num_columns, axis = 1, inplace=True)
            self.data = pd.concat([scaled_data, self.data], axis = 1)
            self.logger.info('Finished Scaling of Numeric Columns')

            return self.data
        
        except Exception as e:
            self.logger.error('Scaling of Numeric Values Unsuccessful: '+ str(e))
            sys.exit(1)


    def resampledata(self, ratio, X_train, y_train):

        try:

            self.logger.info('starting upsampling of data')
            sm = SMOTE(random_state = 0 , n_jobs= -1, ratio = ratio)
            X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)
            X_train_resampled =  pd.DataFrame(X_train_resampled)
            
            return  X_train_resampled, y_train_resampled

        except Exception as e:
            self.logger.error('resampling of data was unsuccessful: '+ str(e))
            sys.exit(1)









            
