"""Various helper routines."""

import numpy as _np


def align_array(arr, dtype, order):
    """
    Make sure that an array is contiguous and aligned with the right type.

    If order='F' use Fortran order. If order='C' use
    C order.

    """

    if order == "F":
        requirements = ["A", "F", "O", "E"]
    elif order == "C":
        requirements = ["A", "C", "O", "E"]
    else:
        raise ValueError("order must be one of 'C' or 'F'.")

    return _np.require(arr, dtype, requirements=requirements)


def assign_parameters(parameters):
    """
    Assigns a parameter object based on input.

    If parameters is None return the global_parameters object.
    Otherewise, return the parameters object again.

    """
    import bempp.api

    if parameters is None:
        new_parameters = bempp.api.GLOBAL_PARAMETERS
    else:
        new_parameters = parameters

    return new_parameters


def promote_to_double_precision(array):
    """Convert an array to real or complex double precision."""

    if array.dtype == "float32":
        return array.astype("float64", copy=False)
    if array.dtype == "complex64":
        return array.astype("complex128", copy=False)
    return array