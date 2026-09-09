import pandas as pd
import numpy as np
from typeguard import typechecked

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
    raise NotImplementedError()

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
    raise NotImplementedError()

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
    raise NotImplementedError()

