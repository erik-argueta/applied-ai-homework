'''
This file is where you migrate all the functions that you made in data_process1 and data_process2. 
'''
import numpy as np
import pandas as pd
import csv
import time # Import the time module
from typeguard import typechecked

# ==========================================================================================================================
# DATA PROCESS 1 FUNCTIONS
# ==========================================================================================================================
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
    num_rows = len(arr)
    column_means = []
    std_devs_array = []
    num_columns = len(arr[0])                                   # len([T, P, TC, SV, Idx])
    focused_columns = arr[:, :num_columns - 1]

    # ==========================
    # Calculate Mean & Std. Dev
    # ==========================
    column_means = np.mean(focused_columns, axis = 0)           # https://numpy.org/devdocs/reference/generated/numpy.mean.html
    std_devs_array = np.std(focused_columns, axis = 0)          # https://numpy.org/devdocs/reference/generated/numpy.std.html

    outlier_cap = np.abs(focused_columns - column_means)
    non_outliers = np.all(outlier_cap <= 2* std_devs_array, axis = 1)
    np_filtered = arr[non_outliers]

    filtered_columns = np_filtered[:, :-1]
    filtered_target = np_filtered[:, -1:]

    column_mins = np.min(filtered_columns, axis = 0)
    column_maxes = np.max(filtered_columns, axis = 0)
    column_ranges = column_maxes - column_mins

    normalized_columns = np.divide(
        filtered_columns - column_mins,
        column_ranges,
        out=np.zeros_like(filtered_columns, dtype=float),
        where=column_ranges != 0
    )

    normalized_arr = np.hstack((normalized_columns, filtered_target))

    if out_file is not None:
        np.savetxt(out_file, normalized_arr, delimiter=',')

    return normalized_arr.shape[0]


# ==========================================================================================================================
# DATA PROCESS 2 FUNCTIONS
# ==========================================================================================================================
@typechecked
def load_dataset_pd(filename: str) -> pd.DataFrame:
    """
    Loads normalized data from a CSV file into a Pandas DataFrame.

    Args:
        filename (str): The path to the CSV file containing normalized data.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the loaded data.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return pd.read_csv(filename, header=None, names=["T", "P", "TC", "SV", "Idx"])

@typechecked
def split_xy(df: pd.DataFrame, y_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a Pandas DataFrame into feature (X) and target (Y) NumPy arrays.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        y_axis (int, optional): The column index of the target variable (Y).
                                Defaults to -1 (last column).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       X (features) and Y (target).

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: A 2-line solution but can be tricky if you are not familiar with pandas.
    """
    X = df.drop(df.columns[y_axis], axis=1).to_numpy()
    Y = df.iloc[:, y_axis].to_numpy()
    
    return X, Y

@typechecked
def split_training_test(
        X_data: np.ndarray, 
        Y_data: np.ndarray, 
        split: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input feature (X) and target (Y) NumPy arrays into training and testing sets.

    Args:
        X_data (np.ndarray): The input feature array.
        Y_data (np.ndarray): The input target array.
        split (float, optional): The proportion of data to be used for training.
                                 Defaults to 0.8 (80% training, 20% testing).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four NumPy arrays:
                                                                X_train, Y_train, X_test, Y_test.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: If you are stuck, look into list slicing. 
    """
    split_index = int(len(X_data) * split)

    X_train = X_data[:split_index]
    Y_train = Y_data[:split_index]
    X_test = X_data[split_index:]
    Y_test = Y_data[split_index:]

    return X_train, Y_train, X_test, Y_test
