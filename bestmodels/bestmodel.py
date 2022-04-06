import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
from src.utils.clustering import Cluster
from src.utils.models import Trainmodel
from evaluation.evalmetrics import Evaluation
import pickle 
from src.utils.common import read_config
from log.logger import Logger
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import json

class Getmodel:

    def __init__(self, data, config_path):

        self.data = data
        self.config_path = config_path
        self.config = read_config(self.config_path)
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def storebestmodel(self, target):

        try:

            self.logger.info('started storing best models for each clusters')
            
            with open('models/kmeans_model.pkl', 'rb') as f:
                k = pickle.load(f)

            val_clusters = k.predict(self.data.drop(target, axis = 1))
            self.data['cluster'] = val_clusters
            unique_cluster = len(self.data['cluster'].unique())

            for cluster in range(0, unique_cluster):

                cluster_val_x =  self.data[self.data['cluster'] == cluster].drop(['cluster','target'], axis = 1)
                cluster_val_y =  self.data[self.data['cluster'] == cluster]['target']
                
                model_dir =  self.config['directory']['model_dir']
                
                for filename in os.scandir(model_dir):
                    
                    if str(cluster) in filename.path:

                        with open(str(filename.path), 'rb') as m:
                            model = pickle.load(m)
                    
    
                        if 'rf' in filename.path:
                
                            prediction = model.predict(cluster_val_x)
                            rf_score = f1_score(cluster_val_y, prediction)
                            evaldata = Evaluation(cluster_val_y, prediction, self.config_path)
                            evaldata.saveconfusionmatrix(cluster, 'rf')
                            evaldata.save_classification_report(cluster, 'rf')
                        
                        elif 'xgb' in filename.path:
                            
                            prediction = model.predict(cluster_val_x)
                            xgb_score = f1_score(cluster_val_y, prediction)
                            evaldata = Evaluation(cluster_val_y, prediction, self.config_path)
                            evaldata.saveconfusionmatrix(cluster, 'xgb')
                            evaldata.save_classification_report(cluster, 'xgb')
                        
                        else:

                            prediction = model.predict(cluster_val_x)
                            catb_score = f1_score(cluster_val_y, prediction)
                            evaldata = Evaluation(cluster_val_y, prediction, self.config_path)
                            evaldata.saveconfusionmatrix(cluster, 'cb')
                            evaldata.save_classification_report(cluster, 'cb')

                if (rf_score >= xgb_score):

                    rf_bestmodel = f'rf_{cluster}_model.pkl'

                    with open("bestmodels/bestmodels.txt", "a") as outfile:
                        outfile.write(rf_bestmodel + '\n')

                else:
                    xgb_bestmodel =  f'xgb_{cluster}_model.pkl'

                    with open("bestmodels/bestmodels.txt", "a") as outfile:
                         outfile.write(xgb_bestmodel + '\n')
                    
        except Exception as e:
            self.logger.error('Saving of best Models for cluster was unsuccessful ', + str(e))   
            sys.exit(1)         
            