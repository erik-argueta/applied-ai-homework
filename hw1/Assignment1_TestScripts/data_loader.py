'''
This file is where you migrate all the functions that you made in data_process1 and data_process2. 
'''
import numpy as np
import pandas as pd
import data_process1 

arr_test = data_process1.load_dataset("hw1/Assignment1_TestScripts/GasProperties.csv")
np_arr = data_process1.load_dataset_np("hw1/Assignment1_TestScripts/GasProperties.csv")
data_process1.normalize_array(arr_test, 'hw1/Assignment1_TestScripts/GasProperties_norm.csv')

# data_process1.normalize_array(arr_test)