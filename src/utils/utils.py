import time
import os 

def get_unique_tablename(file_type):
      unique_name = time.strftime(f"{file_type}_%Y%m%d")
      return unique_name