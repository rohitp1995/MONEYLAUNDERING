import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from src.utils.common import read_config
from log.logger import Logger
from sklearn.cluster import KMeans
import seaborn as sns
from kneed import KneeLocator
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


class Cluster:

    def __init__(self, data, range, config_path):
        self.data = data
        self.range = range
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()
        self.config_path = config_path
        self.config = read_config(self.config_path)

    def save_cluster(self, model):

        try:

            os.makedirs(self.config['directory']['model_dir'], exist_ok = True)
            model_name = f'kmeans_{self.config["names"]["model_name"]}'
            model_dir = os.path.join(self.config['directory']['model_dir'], model_name)
            
            with open(model_dir, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        except Exception as e:
            self.logger.error('Error While saving the cluster: '+ str(e))
            sys.exit(1)
            
    def getclusternumber(self, target):

        try:

            self.wcss = []
            self.logger.info('Finding Optimal cluster number')
            
            for i in range(1, self.range+1):
                k = KMeans(n_clusters=i)
                k.fit(self.data.drop(target,1))
                self.logger.info(f'Finished creating Cluster {i}')
                self.wcss.append(k.inertia_)

            kn = KneeLocator(range(1, self.range+1), self.wcss, curve='convex', direction='decreasing')

            return kn.knee

        except Exception as e:
            self.logger.error('Error While finding cluster numbers: '+ str(e))
            sys.exit(1)

    def saveelbowgraph(self):
        
        try:
            self.logger.info('Generate elbow graph for pre-defined clusters')

            os.makedirs(self.config['directory']['report_dir'], exist_ok = True)
            graph_name = f'kmeans_Elbow_{self.config["names"]["graph_name"]}'
            graph_dir = os.path.join(self.config['directory']['report_dir'], graph_name)

            plt.plot(range(1, self.range+1), self.wcss)
            plt.savefig(graph_dir)

        except Exception as e:
            self.logger.error('Error While saving elbow graph: '+ str(e))
            sys.exit(1)


    def tagclusternumbers(self, cluster, target):
        
        try:
            self.logger.info('Started Tagging of the clusters to dataframe')
            km = KMeans(n_clusters = cluster, random_state = 10)
            
            km.fit(self.data.drop(target,1))
            self.save_cluster(km)
            self.saveelbowgraph()

            self.data['cluster'] =  km.labels_

        except Exception as e:
            self.logger.error('Error While tagging cluster numbers: '+ str(e))
            sys.exit(1)


    def predictcluster(self, x_test, target):
        
        try:
            self.logger.info('Predicting cluster for test data')    
            prediction = km.predict(x_test.drop(target,1))
            x_test['cluster'] =  prediction

        except Exception as e:
            self.logger.error('Error While predicting cluster numbers: '+ str(e))
            sys.exit(1)

    




