import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
from data_ingestion.data_loader_training import TrainingDataGetter
from data_preprocessing.preprocessing import Preprocess
from FeatureGeneration.generateFeatures import GenerateNewFeatures
from src.utils.clustering import Cluster
import argparse 
from src.utils.models import Trainmodel
from bestmodels.bestmodel import getmodel
from DataValidation.TrainingDataValidation import ValidateFile
import pickle

def Training(config_path):
   
   validator = ValidateFile('data_ingestion/Dataset.csv')
   extension_val = validator.validate_extension()
   colsize_val = validator.validate_colsize()
   dtypes_val = validator.validate_coldatatypes()
   
   if all([extension_val, colsize_val, dtypes_val]):

      DataGetter = TrainingDataGetter('data_ingestion/Dataset.csv')
      data = DataGetter.get_data()

      num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                  'newbalanceDest']

      MONEY_OUT = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT']
      MONEY_IN = ['CASH_IN']

      ## creating new features
      featuresgen = GenerateNewFeatures(data)
      featuresgen.make_date_columns('step')
      featuresgen.make_diff_columns(MONEY_IN, MONEY_OUT)

      ## Preprocessing data
      preprocess = Preprocess(data)
      preprocess.drop_columns(['isFlaggedFraud', 'step'])     
      data = preprocess.encode_categorical_columns()
      preprocess.drop_columns(['nameOrig','nameDest'])      
      data = preprocess.scalenumericvalues(num_cols)

      ## splitting and sampling
      X_train, X_val, y_train, y_val = preprocess.split_data(0.3, 'isFraud')
      X_train_resampled, y_train_resampled = preprocess.resampledata(0.5, X_train, y_train)
      
      X_train_resampled['target'] = y_train_resampled
      X_val['target'] = y_val

      # clustering the sampled data
      cluster_obj = Cluster(X_train_resampled, 10, config_path)
      cluster_numbers = cluster_obj.getclusternumber('target')
      cluster_obj.tagclusternumbers(cluster_numbers, 'target')

      # predicting cluster for test data
      cluster_obj.predictcluster(X_val, 'target')

      # with open('models/kmeans_model.pkl', 'rb') as f:
      #    k = pickle.load(f)
      #    clusters = k.predict(X_train_resampled.drop('target',1))
      #    X_train_resampled['cluster'] = clusters

      # cluster_numbers = 4
      for cluster in range(0, cluster_numbers):
         
         cluster_train_x = X_train_resampled[X_train_resampled['cluster'] == cluster].drop(['cluster','target'], axis = 1)
         cluster_train_y = X_train_resampled[X_train_resampled['cluster'] == cluster]['target']

         model_obj = Trainmodel(10, cluster, cluster_train_x, cluster_train_y, config_path)
         rf = model_obj.RandomForestClassifierTrain()
         xgb = model_obj.XGBoostClassifierTrain()

      bestmodel = getmodel(X_val, config_path)
      bestmodel.storebestmodel('target')

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    Training(config_path=parsed_args.config)
    
    

