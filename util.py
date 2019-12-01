"""
UTIL
30 November 2019
Lydia Chan, Russell Tran

Local Module with useful tools
"""
import pandas as pd
from datetime import datetime

class DataSaver():
	def __init__(self, algorithm_name, environment_name):
        now = datetime.now() # current date and time
        date_time_string = now.strftime("%m-%d-%Y-%H%M%S")
        self.csv_name = "{}_{}_{}.csv".format(algorithm_name, environment_name, date_time_string)

    def save_list_of_dicts(self, list_of_dicts):
    	df = pd.DataFrame(list_of_dicts)
    	print("Saving to csv named {}".format(self.csv_name))
    	df.to_csv(self.csv_name)