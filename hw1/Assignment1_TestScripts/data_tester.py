import numpy as np
import pandas as pd
import torch as t
import time
import data_loader


if __name__ == "__main__":
    '''
    You can write your filled out functions in here to test them out.
    '''

    arr_test = data_loader.load_dataset('hw1/Assignment1_TestScripts/GasProperties.csv')
    np_arr = data_loader.load_dataset_np("hw1/Assignment1_TestScripts/GasProperties.csv")
    data_loader.normalize_array(arr_test, 'hw1/Assignment1_TestScripts/GasProperties_norm.csv')
    data_loader.normalize_array_np(np_arr, 'hw1/Assignment1_TestScripts/np_GasProperties_norm.csv')

    pass