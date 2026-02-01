# Helper function: ensure data can be safely passed to MATLAB
import numpy as np


def ensure_contiguous(data, dtype=None):
    """
    Recursively ensure that all NumPy ndarrays are C-contiguous in memory
    and (optionally) cast to a unified dtype.

    Args:
        data: Input data. Can be a dict, list, or np.ndarray. Other types are returned as-is.
        dtype: Target dtype (e.g., np.float64 or 'float64'). If None, preserve the original dtype
               for each ndarray.
    """
    if isinstance(data, np.ndarray):
        # If a target dtype is specified, cast to it; otherwise keep the array's original dtype.
        target_dtype = dtype if dtype is not None else data.dtype

        # np.ascontiguousarray ensures C-contiguous memory layout.
        # astype(..., copy=False) avoids an extra copy when the dtype already matches.
        # Together, this helps meet typical MATLAB/engine interface expectations.
        return np.ascontiguousarray(data.astype(target_dtype, copy=False))

    elif isinstance(data, dict):
        # Recursively process each value in the dictionary.
        return {k: ensure_contiguous(v, dtype) for k, v in data.items()}

    elif isinstance(data, list):
        # Recursively process each element in the list (e.g., a list of position arrays).
        return [ensure_contiguous(i, dtype) for i in data]

    # For non-container, non-ndarray types, return unchanged.
    return data
