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
    # ==========================================
    # Much to refine; very much brute-forced it
    # ==========================================

    max_val = []
    min_val = []
    num_rows = len(arr)
    total_sums = []
    column_means = []
    std_devs = []
    normalized = []

    num_columns = len(arr[0])

    for column_index in range(num_columns - 1):
        column_val = [row[column_index] for row in arr]         # determines column's row value
        max_val.append(max(column_val))                         # appends maximum said value
        min_val.append(min(column_val))                         # appends minimum said value
        total_sums.append(sum(column_val))                      # sums entire column's rows

        mean = sum(column_val) / num_rows
        column_means.append(mean)

        column_max = max(column_val)
        column_min = min(column_val)

        std_devs.append(
            (sum((value - mean)**2 for value in column_val) / num_rows)**0.5           ## lack of math forces 0.5 exponent
        )

        normalized_column = [(value - column_min) / (column_max - column_min) for value in column_val]
        normalized.append(normalized_column)

    # print("max values: ", max_val)
    # print("min values: ", min_val)
    # print("column means", column_means)
    # print("std_dev: ", std_devs)
    # # print("normalized column: ", normalized_column)

    # Export normalized column to file
    with open(out_file, 'w') as f:
        for row in normalized_column:
            f.write(f"{row}\n")

    return num_rows


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

