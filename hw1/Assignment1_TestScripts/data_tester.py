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

    # Question 1(a)
    print("=================")    
    print("| Question 1(a) |")
    print("=================")
    

    # =================
    # Loading CSV
    # =================
    regular_array = data_loader.load_dataset('hw1/Assignment1_TestScripts/GasProperties.csv')
    numpy_dataframe = data_loader.load_dataset_np('hw1/Assignment1_TestScripts/GasProperties.csv')

    # =====================
    # Performance Metrics
    # =====================
    print("  Array Method:")
    start = time.perf_counter()
    data_loader.normalize_array(regular_array, 'hw1/Assignment1_TestScripts/GasProperties_norm.csv')
    array_time = time.perf_counter() - start 
    print(f"  Normalization time: {array_time: .6f}")

    print("\n  Numpy Method:")
    start = time.perf_counter()
    data_loader.normalize_array_np(numpy_dataframe, 'hw1/Assignment1_TestScripts/np_GasProperties_norm.csv')
    numpy_time = time.perf_counter() - start 
    print(f"  Normalization time: {numpy_time: .6f}")


    # Question 1(b)
    print("\n=================")
    print("| Question 1(b) |")
    print("=================")
    pd_data = data_loader.load_dataset_pd('hw1/Assignment1_TestScripts/np_GasProperties_norm.csv')
    X_data, Y_data = data_loader.split_xy(pd_data)

    X_training, Y_training, X_testing, Y_testing = (data_loader.split_training_test(X_data, Y_data))

    print("  X training:", X_training.shape)
    print("  Y training:", Y_training.shape)
    print("  X testing:", X_testing.shape)
    print("  Y testing:", Y_testing.shape)


    print("\n=======================")
    print("| Question 2(a) & (b) |")
    print("=======================")


    X_list = X_data.tolist()
    Y_list = Y_data.tolist()

    start = time.perf_counter()
    py_column = vector_product.find_largest_dot_product_py(X_list, Y_list)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    np_column = vector_product.find_largest_dot_product_np(X_data, Y_data)
    np_time = time.perf_counter() - start 

    column_names = ["T", "P", "TC", "SV"]
    print(f"  Python Largest Column: {column_names[py_column]}")
    print(f"  Python time: {py_time:.6f} seconds.")
    print(f"  NumPy Largest Column: {column_names[np_column]}")
    print(f"  NumPy time: {np_time:.6f} seconds.")
    print(f"  NumPy speedup: x{py_time / np_time:.2f}")


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