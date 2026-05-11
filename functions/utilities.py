import numpy as np

def gauss(x, amplitude, mu, sigma):
    """
    Gaussian profile used to refine the position and width of significant peaks.
    """
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
