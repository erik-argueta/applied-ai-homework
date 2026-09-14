import numpy as np
import pandas as pd
import torch as t
import time
import data_loader
import vector_product


if __name__ == "__main__":
    '''
    You can write your filled out functions in here to test them out.
    '''

    # Main Driver
    dataframe = data_loader.load_dataset_pd('hw1/Assignment1_TestScripts/GasProperties.csv')
    X_data, Y_data = data_loader.split_xy(dataframe)

    X_list = X_data.tolist()
    Y_list = Y_data.tolist()

    start = time.perf_counter()
    py_column = vector_product.find_largest_dot_product_py(X_list, Y_list)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    np_column = vector_product.find_largest_dot_product_np(X_data, Y_data)
    np_time = time.perf_counter() - start 

    print(f"Largest Column: {py_column}")
    print(f"Python time: {py_time:.6f} seconds.")
    print(f"NumPy Largest Column: {np_column}")
    print(f"Python time: {np_time:.6f} seconds.")
    print(f"NumPy speedup: {py_time / np_time:.2f}x")


    # Matrix Multiplication
    X32 = X_data.astype(np.float32)
    X64 = X_data.astype(np.float64)

    start = time.perf_counter()
    result32 = vector_product.mat_mul_np(X32.T, X32)
    time32 = time.perf_counter() - start 

    start = time.perf_counter()
    result64 = vector_product.mat_mul_np(X64.T, X64)
    time64 = time.perf_counter() - start

    print(f"float32 matrix time: {time32:.6f} seconds")
    print(f"float64 matrix time: {time64:.6f} seconds")

    print(f"Result shape: {result32.shape}")