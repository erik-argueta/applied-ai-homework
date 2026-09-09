import numpy as np
import torch as t
from typeguard import typechecked

@typechecked
def dot_product(a: list[float], b: list[float]) -> float:
    """
    Calculates the dot product of two lists of floats.

    Args:
        a (list[float]): The first list of float values.
        b (list[float]): The second list of float values.

    Returns:
        float: The dot product of the two lists.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: You can use standard for-loops or list comprehension if you want to be fancy.
    If you want a challenge, try writing this in 1 line.
    """
    raise NotImplementedError()

@typechecked
def find_largest_dot_product_py(X_data: list[list[float]], Y_data: list[float]) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the pure Python implementation.

    Args:
        X_data (list[list[float]]): A list of lists representing the input features.
        Y_data (list[float]): A list representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: I recommend using the dot_product function you wrote above to complete this.
    The implementation for this function is pretty straightforward.
    """
    raise NotImplementedError()

@typechecked
def dot_product_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the dot product of two NumPy arrays.

    Args:
        a (np.ndarray): The first NumPy array.
        b (np.ndarray): The second NumPy array.

    Returns:
        float: The dot product of the two arrays.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    raise NotImplementedError()

@typechecked
def find_largest_dot_product_np(X_data: np.ndarray, Y_data: np.ndarray) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the NumPy implementation.

    Args:
        X_data (np.ndarray): A NumPy array representing the input features.
        Y_data (np.ndarray): A NumPy array representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This might be a little more tricky to do, the solution itself is pretty short (3 lines) 
    but finding the right function in numpy might be difficult.
    """
    raise NotImplementedError()

@typechecked
def mat_mul_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs matrix multiplication on two NumPy arrays.

    Args:
        A (np.ndarray): The first NumPy array (matrix).
        B (np.ndarray): The second NumPy array (matrix).

    Returns:
        np.ndarray: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    raise NotImplementedError()

@typechecked
def mat_mul_t(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """
    Performs matrix multiplication on two PyTorch tensors.

    Args:
        A (t.Tensor): The first PyTorch tensor (matrix).
        B (t.Tensor): The second PyTorch tensor (matrix).

    Returns:
        t.Tensor: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    raise NotImplementedError()

@typechecked
def dot_product_t(a: t.Tensor, b: t.Tensor) -> t.tensor:
    """
    Calculates the dot product of two PyTorch tensors.

    Args:
        a (t.Tensor): The first PyTorch tensor.
        b (t.Tensor): The second PyTorch tensor.

    Returns:
        t.Tensor: The dot product of the two tensors.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    raise NotImplementedError()
