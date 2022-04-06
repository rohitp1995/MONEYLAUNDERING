import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(CURRENT_DIR)))
from src.utils.common import read_config
from log.logger import Logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pickle

class Trainmodel:

    def __init__(self, cv, cluster_number, X_train, y_train, config_path):
        
        self.cv = cv
        self.cluster_number = cluster_number
        self.X_train = X_train
        self.y_train = y_train
        self.config_path = config_path
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()
        self.config = read_config(self.config_path)

    def RandomForestClassifierTrain(self):

        try:
            model = RandomForestClassifier()
            random_search = RandomizedSearchCV(model, param_distributions = self.config['randomforestparams'], cv = self.cv, scoring='f1', n_jobs = 3, verbose = 20, n_iter = 50)
            self.logger.info(f'Started Training RandomForest Model on Cluster {self.cluster_number}')
            random_search.fit(self.X_train, self.y_train)
            self.logger.info(f'Finished Training RandomForest Model on Cluster {self.cluster_number}')
            rf = random_search.best_estimator_
            rf.fit(self.X_train, self.y_train)
            
            os.makedirs(self.config['directory']['model_dir'], exist_ok = True)
            model_name = f'rf_{self.cluster_number}_{self.config["names"]["model_name"]}'
            model_dir = os.path.join(self.config['directory']['model_dir'], model_name)

            with open(model_dir, 'wb') as handle:
                pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)


        except Exception as e:
            self.logger.error('RandomForest training is Unsuccessful: '+ str(e))
            sys.exit(1)


    def XGBoostClassifierTrain(self):

        try:
            model = XGBClassifier()
            random_search = RandomizedSearchCV(model, param_distributions = self.config['xgboostparams'], cv = self.cv, verbose = 20, scoring='f1', n_iter = 50)
            self.logger.info(f'Started Training XGboost Model on Cluster {self.cluster_number}')
            random_search.fit(self.X_train, self.y_train, eval_metric = 'logloss')
            self.logger.info(f'Finished Training XGboost Model on Cluster {self.cluster_number}')
            xgb = random_search.best_estimator_
            xgb.fit(self.X_train, self.y_train)

            os.makedirs(self.config['directory']['model_dir'], exist_ok=True)
            model_name = f'xgb_{self.cluster_number}_{self.config["names"]["model_name"]}'
            model_dir = os.path.join(self.config["directory"]["model_dir"], model_name)

            with open(model_dir, 'wb') as handle:
                pickle.dump(xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        except Exception as e:
            self.logger.error('XGboost training is Unsuccessful: '+ str(e))
            sys.exit(1)


    # def CatboostClassifierTrain(self):

    # try:
    #     self.model = CatBoostClassifier()
    #     self.random_search = RandomizedSearchCV(self.model, param_distributions = self.config['catboostparams'], cv = self.cv, scoring='f1', n_jobs = -1)
    #     self.logger.info(f'Started Training Catboost Model on Cluster {self.cluster_number}')
    #     self.random_search.fit(self.X_train, self.y_train)
    #     self.logger.info(f'Finished Training Catboost Model on Cluster {self.cluster_number}')
    #     self.cb = random_search.best_estimator_
    #     self.cb.fits(self.X_train, self.y_train)

    #     self.model_name = f'cb_{cluster_number}_{config['directory']['model_name']}
    #     self.model_dir = os.path.join(config['directory']['model_dir'], self.model_name)

    #     with open(self.model_dir, 'wb') as handle:
    #         pickle.dump(self.xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     return self.rf
    
    # except Exception as e:
    #     self.logger.error('Catboost training is Unsuccessful'+ str(e))





    


            
            