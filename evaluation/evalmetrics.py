import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from src.utils.common import read_config
from log.logger import Logger
from sklearn.metrics import confusion_matrix,classification_report

class Evaluation:
    def __init__(self, y, y_pred, config_path):

        self.y = y
        self.y_pred = y_pred
        self.config_path = config_path
        self.config = read_config(self.config_path)
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def saveconfusionmatrix(self, cluster, model_alias):

        cm = confusion_matrix(self.y, self.y_pred)
        
        plt.figure()
        image = sns.heatmap(cm, annot=True)
        
        os.makedirs(self.config['directory']['report_dir'], exist_ok = True)
        graph_name = f'{model_alias}_confusion_matrix_{cluster}_{self.config["names"]["graph_name"]}'
        graph_dir = os.path.join(self.config['directory']['report_dir'], graph_name)
        
        plt.savefig(graph_dir)

    def save_classification_report(self, cluster, model_alias):
       
        report = classification_report(self.y, self.y_pred, output_dict=True)
       
        df_classification_report = pd.DataFrame(report).transpose()
        df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)

        os.makedirs(self.config['directory']['report_dir'], exist_ok = True)
        report_name = f'{model_alias}_{cluster}_{self.config["names"]["report_name"]}'
        file_dir = os.path.join(self.config['directory']['report_dir'], report_name)

        df_classification_report.to_excel(file_dir, engine = 'openpyxl')

