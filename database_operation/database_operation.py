import mysql.connector as sql
import logging
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
from log.logger import Logger


class DB_Operations:
    
    def __init__(self, db_name, host, username, pwd):
        self.db_name = db_name
        self.host = host
        self.username = username
        self.pwd = pwd
        self.log_obj = Logger('Generatedlogs')
        self.logger = self.log_obj.logging()

    def connect_db(self):
        
        try:
            self.db = sql.connect(host=self.host, database=self.db_name, user=self.username, password=self.pwd)
            self.logger.info(f"Connected to the { self.db_name}")
        except Exception as e: 
            self.logger.error(f"There was some error while Connecting to the {self.db_name} database: " + str(e))

    def checkTableExists(self, tablename):
        
        try:
            if self.db.is_connected():
                curs = self.db.cursor()
                curs.execute(f'SELECT COUNT(*) FROM information_schema.tables WHERE table_name = {tablename}')
                if dbcur.fetchone()[0] == 1:
                    curs.close()
                    return True
                curs.close()
                return False
            else:
                self.logger.error(f"There was some error while Connecting to the {self.db_name} database")
        except Exception as e: 
            self.logger.error(f"There was some error while checking  for {tablename} existence: " + str(e))


    def get_data_from_table(self,table_name):
        
        try:
            if self.db.is_connected():
                 data = pd.read_sql(f'select * from {table_name}', con = self.db)
            else:
                self.logger.info(f"There was some error while Connecting to the {self.db_name} database: ")
        except Exception as e:
             self.logger.error(f"There was some error while fetching data from table: " + str(e))
        
        return data

    def create_table_for_training(self, table_name):

        try:
            if self.db.is_connected():
                if self.checkTableExists(table_name):
                    self.logger.info(f"Table {table_name} already exists")
                else:
                    create_table_query = f""" Create table {table_name} as select * from dataset where 1=2 """
                    curs = self.db.cursor()
                    curs.execute(create_table_query)
                    self.logger.info(f"Table {table_name} created for training data")
        except Exception as e:
             self.logger.error(f"There was some error while creating {table_name} table: " + str(e))


    def create_table_for_Prediciton(self, table_name):
        
        try:
            if self.db.is_connected():
                if self.checkTableExists(table_name):
                    self.logger.info(f"Table {table_name} already exists")
                else:          
                    create_table_query = f""" Create table {table_name} as select * from dataset where 1 = 2 """
                    curs = self.db.cursor()
                    cursor.execute(create_table_query)
                    self.logger.info(f"Table {table_name} created for Prediction data")
        except Exception as e:
            self.logger.error(f"There was some error while creating {table_name} table: " + str(e))


    def InsertTrainingData(self, table_name, values):
        
        try:
            if self.db.is_connected():
                if self.checkTableExists(table_name):
                    create_insert_query = f""" Insert into {table_name} values ({values}) """
                    curs = self.db.cursor()
                    cursor.execute(create_insert_query)
                    self.logger.info(f"inserted  data in {table_name} for Training data")
                else:
                    self.logger.info(f"There is no table named as {table_name}")
        except Exception as e:
             self.logger.error(f"There was some error while inserting data in {table_name} table: " + str(e))



    def InsertPredictionData(self,table_name):

        try:
            if self.db.is_connected():
                if self.checkTableExists(table_name):
                    create_insert_query = f""" Insert into {table_name} values ({values}) """
                    curs = self.db.cursor()
                    cursor.execute(create_insert_query)
                    self.logger.info(f"inserted  data in {table_name} for Prediction data")
                else:
                    self.logger.info(f"There is no table named as {table_name}")
        except Exception as e:
             self.logger.error(f"There was some error while inserting data in {table_name} table: " + str(e))


