import numpy as np


def model(x,
          *params,
          polynomial_degree=2,
          include_slow_harmonics=True,
          base_period_slow_harmonics=30,
          slow_harmonics=None):
    """
    Compute the model value at each point x.

    The model includes:
    - A polynomial trend of degree 1, 2, or 3
    - Optional slow harmonics (bLk, cLk) with configurable base period and orders
    - The first seasonal harmonic (b1, c1) with time-varying amplitude through b1' and c1'
    - Seasonal harmonics 2 to 4 with fixed amplitude

    Parameters
    ----------
    x : array-like
        Centered dates.
    polynomial_degree : int, optional
        Degree of the polynomial trend. Allowed values are 1, 2, and 3.
    include_slow_harmonics : bool, optional
        Whether to include the slow harmonics.
    base_period_slow_harmonics : float, optional
        Base period (in years) for the slow harmonics.
    slow_harmonics : list, optional
        Harmonic orders included in the slow component.
    params : tuple
        Model parameters, in the expected order.

    Returns
    -------
    np.ndarray
        Model values evaluated at x.
    """
    if slow_harmonics is None:
        slow_harmonics = []

    if polynomial_degree not in [1, 2, 3]:
        raise ValueError("polynomial_degree must be 1, 2, or 3")

    idx = 0

    # 1) Polynomial
    n_poly_params = polynomial_degree + 1
    polynomial_coeffs = params[idx:idx + n_poly_params]
    poly = np.zeros_like(x, dtype=float)

    for i, coeff in enumerate(polynomial_coeffs):
        poly += coeff * x**i

    idx += n_poly_params

    # 2) Selected slow harmonics
    slow_harm = 0.0
    if include_slow_harmonics and len(slow_harmonics) > 0:
        for k in slow_harmonics:
            bL = params[idx]
            cL = params[idx + 1]
            omega = 2 * np.pi * k / base_period_slow_harmonics
            slow_harm += bL * np.sin(omega * x) + cL * np.cos(omega * x)
            idx += 2

    # 3) First harmonic with time-varying amplitude
    b1, c1, bp1, cp1 = params[idx:idx + 4]
    harmonic1 = ((b1 + bp1 * x) * np.sin(2 * np.pi * x) +
                 (c1 + cp1 * x) * np.cos(2 * np.pi * x))
    idx += 4

    # 4) Fixed harmonics 2-4
    harmonic_rest = 0.0
    for k in range(2, 5):
        b = params[idx]
        c = params[idx + 1]
        omega = 2 * np.pi * k
        harmonic_rest += b * np.sin(omega * x) + c * np.cos(omega * x)
        idx += 2

    return poly + slow_harm + harmonic1 + harmonic_rest


def model_components(x,
                     *params,
                     polynomial_degree=2,
                     include_slow_harmonics=True,
                     base_period_slow_harmonics=30,
                     slow_harmonics=None,
                     return_dlf=False):
    """
    Compute the value of each model component at each point x.

    Components
    ----------
    poly : polynomial trend
        Polynomial trend of degree 1, 2, or 3.
    seasonal : seasonal component
        First harmonic with time-varying amplitude plus fixed harmonics 2-4.
    lf : low-frequency component
        Selected slow harmonics.
    dlf : low-frequency derivative, optional
        Time derivative of lf, returned only if return_dlf=True.
    """
    if slow_harmonics is None:
        slow_harmonics = []

    if polynomial_degree not in [1, 2, 3]:
        raise ValueError("polynomial_degree must be 1, 2, or 3")

    idx = 0

    # 1) Polynomial
    n_poly_params = polynomial_degree + 1
    polynomial_coeffs = params[idx:idx + n_poly_params]
    poly = np.zeros_like(x, dtype=float)

    for i, coeff in enumerate(polynomial_coeffs):
        poly += coeff * x**i

    idx += n_poly_params

    # 2) Selected slow harmonics
    lf = 0.0
    dlf = 0.0
    if include_slow_harmonics and len(slow_harmonics) > 0:
        for k in slow_harmonics:
            bL = params[idx]
            cL = params[idx + 1]
            omegaL = 2 * np.pi * k / base_period_slow_harmonics
            lf += bL * np.sin(omegaL * x) + cL * np.cos(omegaL * x)
            if return_dlf:
                dlf += bL * omegaL * np.cos(omegaL * x) - cL * omegaL * np.sin(omegaL * x)
            idx += 2

    # 3) First harmonic with time-varying amplitude
    b1, c1, bp1, cp1 = params[idx:idx + 4]
    harmonic1 = ((b1 + bp1 * x) * np.sin(2 * np.pi * x) +
                 (c1 + cp1 * x) * np.cos(2 * np.pi * x))
    idx += 4

    # 4) Fixed harmonics 2-4
    harmonic_rest = 0.0
    for k in range(2, 5):
        b = params[idx]
        c = params[idx + 1]
        omega = 2 * np.pi * k
        harmonic_rest += b * np.sin(omega * x) + c * np.cos(omega * x)
        idx += 2

    seasonal = harmonic1 + harmonic_rest

    if return_dlf:
        return poly, seasonal, lf, dlf
    return poly, seasonal, lf