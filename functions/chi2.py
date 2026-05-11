import numpy as np


def calculate_chi2(observed_means, observed_stds, y_fit, n_parameters):
    """
    Compute chi2, degrees of freedom (dof), and reduced chi2.
    The y_fit array must be evaluated at the observation dates.
    """
    residuals = (observed_means - y_fit) / observed_stds
    chi2 = np.sum(residuals**2)
    dof = len(observed_means) - n_parameters

    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    reduced_chi2 = chi2 / dof

    return chi2, dof, reduced_chi2