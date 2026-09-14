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
    regular_array = data_loader.load_dataset('GasProperties.csv')
    numpy_dataframe = data_loader.load_dataset_np('GasProperties.csv')

    # =====================
    # Performance Metrics
    # =====================
    print("  Array Method:")
    start = time.perf_counter()
    data_loader.normalize_array(regular_array, 'GasProperties_norm.csv')
    array_time = time.perf_counter() - start 
    print(f"  Normalization time: {array_time: .6f} seconds")

    print("\n  Numpy Method:")
    start = time.perf_counter()
    data_loader.normalize_array_np(numpy_dataframe, 'np_GasProperties_norm.csv')
    numpy_time = time.perf_counter() - start 
    print(f"  Normalization time: {numpy_time: .6f} seconds")
    print(f"\n  Time difference: {abs(numpy_time-array_time): .6f} seconds")


    # Question 1(b)
    print("\n=================")
    print("| Question 1(b) |")
    print("=================")
    pd_data = data_loader.load_dataset_pd('np_GasProperties_norm.csv')
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
    print(f"  NumPy speedup: {py_time / np_time:.2f}x")
    print(f"  Time difference: {abs(py_time - np_time): .6f} seconds.")


    print("\n=======================")
    print("| Question 2(c) & (d) |")
    print("=======================")

    X32 = X_data.astype(np.float32)
    X64 = X_data.astype(np.float64)

    start = time.perf_counter()
    result32 = vector_product.mat_mul_np(X32.T, X32)
    time32 = time.perf_counter() - start 

    start = time.perf_counter()
    result64 = vector_product.mat_mul_np(X64.T, X64)
    time64 = time.perf_counter() - start

    print(f"  float32 matrix time: {time32:.6f} seconds")
    print(f"  float64 matrix time: {time64:.6f} seconds")
    print(f"  Time difference: {abs(time32-time64): .6f} seconds")

    print(f"  Result shape: {result32.shape}")

    '''
    AI Assisted Code to compare GPU and CPU
    '''
    if not t.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available")
    
    device = t.device("cuda:0")
    repeats = 50
    
    # Create identical 64-bit tensors.
    X_cpu = t.as_tensor(X_data, dtype=t.float64, device="cpu")
    X_gpu = X_cpu.to(device)

    print("\n=========================")
    print("| GPU and CPU Comparison| ")
    print("=========================")
    
    print("  GPU:", t.cuda.get_device_name(0))
    print("  Data type:", X_cpu.dtype)
    print("  Input shape:", tuple(X_cpu.shape))
    
    # X_data is rectangular, so calculate X.T @ X.
    # Warm up CPU and GPU before measuring.
    for _ in range(5):
        cpu_result = vector_product.mat_mul_t(X_cpu.T, X_cpu)
        gpu_result = vector_product.mat_mul_t(X_gpu.T, X_gpu)
    
    t.cuda.synchronize()
    
    # CPU timing
    start = time.perf_counter()
    
    for _ in range(repeats):
        cpu_result = vector_product.mat_mul_t(X_cpu.T, X_cpu)
    
    cpu_time = (time.perf_counter() - start) / repeats
    
    # GPU timing: CUDA events account for asynchronous execution.
    gpu_start = t.cuda.Event(enable_timing=True)
    gpu_end = t.cuda.Event(enable_timing=True)
    
    t.cuda.synchronize()
    gpu_start.record()
    
    for _ in range(repeats):
        gpu_result = vector_product.mat_mul_t(X_gpu.T, X_gpu)
    
    gpu_end.record()
    t.cuda.synchronize()
    
    gpu_time = gpu_start.elapsed_time(gpu_end) / 1000 / repeats
    
    # Confirm both devices produced equivalent results.
    t.testing.assert_close(
        cpu_result,
        gpu_result.cpu(),
        rtol=1e-10,
        atol=1e-10,
    )
    
    # X.T @ X requires approximately 2*m*n*k operations.
    m, k = X_cpu.T.shape
    _, n = X_cpu.shape
    operation_flops = 2 * m * n * k
    
    print(f"  CPU average time: {cpu_time:.9f} seconds")
    print(f"  GPU average time: {gpu_time:.9f} seconds")
    print(f"  Time difference: {abs(cpu_time-gpu_time): .6f} seconds.")
    print(f"  GPU speedup: {cpu_time / gpu_time:.2f}x")
    print(f"  CPU performance: {operation_flops / cpu_time / 1e9:.3f} GFLOPS")
    print(f"  GPU performance: {operation_flops / gpu_time / 1e9:.3f} GFLOPS")