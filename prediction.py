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
import pickle

def prediction(config_path):
   
   DataGetter = TrainingDataGetter('data_ingestion/Dataset.csv')
   data = DataGetter.get_data()

   preprocess = Preprocess(data)
   preprocess.drop_columns('isFlaggedFraud')
    
   featuresgen = GenerateNewFeatures(data)
   featuresgen.make_date_columns('step')
     
   data = preprocess.encode_categorical_columns()
    
   preprocess.drop_columns(['nameOrig','nameDest'])  

   num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
               'newbalanceDest']
    
   data = preprocess.scalenumericvalues(num_cols)
   

   
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    Training(config_path=parsed_args.config)
    
    

