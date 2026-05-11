import numpy as np

def transform_x1987_to_x2019(values, scale, m_coeff, n_coeff):
    """
    Transform data in the x1987 scale (scale == 61 in the WDCGG metadata) to X2019,
    using the relation CO2_X2019 = m_coeff * CO2_X1987 + n_coeff.

    The input must be ordered arrays with values and their corresponding scale,
    together with the coefficients of the relation.

    Return transformed values, modifying only those in the x1987 scale
    (scale == 61) and leaving the rest unchanged.
    """
    transformed_values = np.array(values, dtype=float, copy=True)
    mask = (scale == 61)
    transformed_values[mask] = m_coeff * transformed_values[mask] + n_coeff
    return transformed_values


def transform_x1993_to_x2019(values, scale, m_coeff, n_coeff):
    """
    Transform data in the x1993 scale (scale == 59 in the WDCGG metadata) to X2019,
    using the relation CO2_X2019 = m_coeff * CO2_X1993 + n_coeff.

    The input must be ordered arrays with values and their corresponding scale,
    together with the coefficients of the relation.

    Return transformed values, modifying only those in the x1993 scale
    (scale == 59) and leaving the rest unchanged.
    """
    transformed_values = np.array(values, dtype=float, copy=True)
    mask = (scale == 59)
    transformed_values[mask] = m_coeff * transformed_values[mask] + n_coeff
    return transformed_values


def increase_x1987_errors(errors, scale, added_error):
    """
    Increase the errors of data in the x1987 scale (scale == 61 in the WDCGG metadata)
    by an amount added_error.

    The input must be ordered arrays with errors and their corresponding scale,
    together with a float giving the value of added_error.

    Return transformed errors, modifying only those in the x1987 scale
    (scale == 61) and leaving the rest unchanged.
    """
    transformed_errors = np.array(errors, dtype=float, copy=True)
    mask = (scale == 61)
    transformed_errors[mask] = transformed_errors[mask] + added_error
    return transformed_errors


def increase_x1993_errors(errors, scale, added_error):
    """
    Increase the errors of data in the x1993 scale (scale == 59 in the WDCGG metadata)
    by an amount added_error.

    The input must be ordered arrays with errors and their corresponding scale,
    together with a float giving the value of added_error.

    Return transformed errors, modifying only those in the x1993 scale
    (scale == 59) and leaving the rest unchanged.
    """
    transformed_errors = np.array(errors, dtype=float, copy=True)
    mask = (scale == 59)
    transformed_errors[mask] = transformed_errors[mask] + added_error
    return transformed_errors