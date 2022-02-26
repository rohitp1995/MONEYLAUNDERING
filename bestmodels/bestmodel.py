import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
from src.utils.clustering import Cluster
import argparse 
from src.utils.models import Trainmodel
import pickle 
from src.utils.common import read_config
from log.logger import Logger
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import json

class getmodel:

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
                self.k = pickle.load(f)

            self.val_clusters = self.k.predict(self.data.drop(target, axis = 1))
            self.data['cluster'] = self.val_clusters
            self.unique_cluster = len(self.data['cluster'].unique())

            for cluster in range(0, self.unique_cluster):

                self.cluster_val_x =  self.data[self.data['cluster'] == cluster].drop(['cluster','target'], axis = 1)
                self.cluster_val_y = self.data[self.data['cluster'] == cluster]['target']
                
                self.model_dir =  self.config['directory']['model_dir']
                
                for filename in os.scandir(self.model_dir):
                    
                    if str(cluster) in filename.path:

                        with open(str(filename.path), 'rb') as m:
                            self.model = pickle.load(m)
    
                        if 'rf' in filename.path:
                
                            self.prediction = self.model.predict(self.cluster_val_x)
                            self.rf_score = f1_score(self.cluster_val_y, self.prediction)
                        
                        elif 'xgb' in filename.path:
                            self.prediction = self.model.predict(self.cluster_val_x)
                            self.xgb_score = f1_score(self.cluster_val_y, self.prediction)
                        
                        else:
                            self.prediction = self.model.predict(self.cluster_val_x)
                            self.catb_score = f1_score(self.cluster_val_y, self.prediction)


                if (self.rf_score >= self.xgb_score):

                    self.bestmodel_dict = {'cluster': cluster,
                                        'model': f'rf_{cluster}_model.pkl'}

                    with open("bestmodels/bestmodels.json", "a") as outfile:
                        json.dump(self.bestmodel_dict, outfile)

                else:
                    self.bestmodel_dict = {'cluster': cluster,
                                        'model': f'xgb_{cluster}_model.pkl'}

                    with open("bestmodels/bestmodels.json", "a") as outfile:
                        json.dump(self.bestmodel_dict, outfile)

                # if (self.rf_score > self.xgb_score) and (self.rf_score > self.catb_score):

                #     self.bestmodel_dict = {'cluster': cluster,
                #                         'model': f'rf_{cluster}_model.pkl'}

                    # with open("bestmodels.json", "a") as outfile:
                    #    json.dump(self.bestmodel_dict, outfile)

                # elif (self.xgb_score > self.rf_score) and (self.xgb_score > self.catb_score):

                #     self.bestmodel_dict = {'cluster': cluster,
                #                             'model': f'xgb_{cluster}_model.pkl'}

                #     with open("bestmodels.json", "a") as outfile:
                #         json.dump(self.bestmodel_dict, outfile)

                # else:
                #     self.bestmodel_dict = {'cluster': cluster,
                #                             'model': f'cb_{cluster}_model.pkl'}

                #     with open("bestmodels.json", "a") as outfile:
                #         json.dump(self.bestmodel_dict, outfile)
                    
        except Exception as e:
            self.logger.error('Saving of best Models for cluster was unsuccessful', + str(e))            
            