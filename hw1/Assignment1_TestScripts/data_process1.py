import numpy as np
import pandas as pd
import csv
import time # Import the time module
from typeguard import typechecked

@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list[list[float]]: A list of lists, where each inner list represents a row
                           of the dataset with float values.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: If you are stuck on this, I recommend looking through the python csv library.
    """
    GasProperties = []

    with open(filename, "r") as file:
        reader = csv.reader(file)                                       # establishes reader with provided filepath
        next(reader)                                                    # since first row is filled with strings, we skip first row

        for row in reader:                                              # for every row in the file
            GasProperties.append([float(value) for value in row])       # append to data: float-converted value for each value within the row

    return GasProperties 

@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    """
    Loads a dataset from a CSV file into a NumPy array.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array representing the dataset.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Numpy has a very useful csv file reader called genfromtxt.
    """
    GasProperties2 = np.genfromtxt(filename, dtype=float, skip_header=1, delimiter=',') # skip_header to skip initial row, delimiter to seperate by commas
    return GasProperties2


@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    """
    Normalizes the input array (list of lists) using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (list[list[float]]): The input dataset as a list of lists of floats.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Probably the most complicated function to write because you cant use numpy.
    I would spend some time on this to make sure that all the equations for metrics are correct.
    """
    # =================================
    # Begin with declaring
    # ================================
    num_rows = len(arr)
    column_means = []
    std_devs_array = []
    num_columns = len(arr[0])                                   # len([T, P, TC, SV, Idx]) 
    focused_columns = num_columns - 1                           # columns excluding Idx (4)


    # ==========================
    # Mean & Standard Deviation
    # ==========================
    for column_index in range(focused_columns):                 # for index in (1, 2, 3, 4)
        column_val = [row[column_index] for row in arr]         # AI assist - establishes column values as 2D array containing focused columns 
            # [T0, P0, TC0, SV0]
            # [T1, P1, TC1, SV1]
            # [T2, P2, TC2, SV2]
                                                                 
        mean = sum(column_val) / num_rows                       # w/in same iteration, the written column is summed and divided by rows 
        column_means.append(mean)

        std_devs_array.append(
            (sum((value - mean) ** 2 for value in column_val) / num_rows)**0.5      # lack of math library
        )



    # ======================
    # Filtering Outliers
    # ======================
    filtered_rows = []

    for row in arr:
        outlier = False                                         # outlier is set to not present

        for column_index in range(focused_columns):
            mean = column_means[column_index]                   # mean is recalled from column_means by index
            std_dev = std_devs_array[column_index]              # std dev is recalled from respective array by index

            if (abs(row[column_index] - mean) > 2 * std_dev):
                outlier = True                                  # outlier detected
                break 

        if outlier == False:
            filtered_rows.append(row)



    # =====================
    # Manual Normalization
    # =====================

    normalized_rows = []
    column_mins = []
    column_maxes = []

    for column_index in range(focused_columns):
        column_val = [filtered_row[column_index] for filtered_row in filtered_rows]     # filtered_rows[c_index] row y row
        column_mins.append(min(column_val))
        column_maxes.append(max(column_val))

    for row in filtered_rows:
        normalized_row = []

        for column_index in range(focused_columns):
            column_max = column_maxes[column_index]
            column_min = column_mins[column_index]

            if column_max == column_mins:
                normalized_rows.append(0.0)
            else:
                normalized_row.append((row[column_index] - column_min) / (column_max - column_min))

        normalized_row.append(row[-1])
        normalized_rows.append(normalized_row)

    # AI assist with writing out rows
    if out_file is not None:
        with open(out_file, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows(normalized_rows)

    return len(normalized_rows)
        


@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    """
    Normalizes the input NumPy array using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (np.ndarray): The input dataset as a NumPy array.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: The same function as normalize_array but using numpy to calculate the metrics.
    This function should be almost copy and paste with numpy functions.
    """
    raise NotImplementedError()

