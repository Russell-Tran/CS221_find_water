"""
UTIL
30 November 2019
Lydia Chan, Russell Tran

Local Module with useful tools
"""
import pandas as pd
from datetime import datetime
import os

class DataSaver():
    """
    Wrapper class to make it more convenient to backup data to CSV constantly.
    NOTE: Be sure not to open the CSV while code is working with this class
    """
    def __init__(self, algorithm_name, environment_name, out_path):
        now = datetime.now() # current date and time
        date_time_string = now.strftime("%m-%d-%Y-%H%M%S")
        self.csv_name = "{}{}_{}_{}.csv".format(out_path,algorithm_name, environment_name, date_time_string)

    def save_list_of_dicts(self, list_of_dicts):
        name = self.csv_name
        df = pd.DataFrame(list_of_dicts)
        print("Saving to csv named {}".format(self.csv_name))
        with open(name, "w+") as f:           
             df.to_csv(name)