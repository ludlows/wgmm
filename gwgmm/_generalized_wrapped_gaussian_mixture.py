# Generalized Wrapped Gaussian Mixture Model
# author: https://github.com/ludlows
# 2021-Oct


import numpy as np
from numpy.lib.shape_base import expand_dims


def _unwrap(values, periods):
    """unwrap values using corresponding periods
    
    Parameters
    ----------
    values : array-like of shape (n,)
        The proportions of components of each mixture.
    periods : array-like of shape (n,)
        The periods of values
    
    Returns
    -------
    unwrapped : array, shape (n,)
    """
    return np.fmod(values, periods) - 0.5 * periods



def _gernalized_wrapped_gaussian_dist_exp(x, mu, sigma, periods):
    """exponential term of probability density function (PDF) in generalized wrapped gaussian distribution
   
    Parameters
    ----------
    x : array-like of shape (n,)
        The input vector.
    mu : array-like of shape (n,)
        The mean vector.
    sigma : array-like of shape (n,n)
        The covariance matrix.
    periods : array-like of shape (n,)

    Returns
    -------
    pdf : scalar
    """
    pdf = 0.0
    period_indices = np.expand_dims(np.array([-1, 0, 1]), 0)
    t = np.prod(np.expand_dims(periods, 1), period_indices)
    difference = x - t - mu
    sigma_inv = np.inv(sigma)
    for i in range(3):
        diff = difference[:,i,None]
        exp_term = np.exp(-0.5 * diff.T * sigma_inv * diff)
        pdf += exp_term
    return pdf


def _gernalized_wrapped_gaussian_dist_full(x, mu, sigma, periods):
    """probability density function (PDF) in generalized wrapped gaussian distribution
   
    Parameters
    ----------
    x : array-like of shape (n,)
        The input vector.
    mu : array-like of shape (n,)
        The mean vector.
    sigma : array-like of shape (n,n)
        The covariance matrix.
    periods : array-like of shape (n,)
        The periods of corresponding dimensions in x

    Returns
    -------
    pdf : scalar
    """
    k = len(x)
    exp_term = _gernalized_wrapped_gaussian_dist_exp(x, mu, sigma, periods)
    return 1.0 / np.sqrt((np.pi*2)**k * np.linalg.det(sigma)) * exp_term


def _check_periods(periods):
    """check if all the periods values are positive.
   
    Parameters
    ----------
    periods : array-like of shape (n,)
        The periods of corresponding dimensions in x

    Returns
    -------
    valid : scalar, boolean
    """
    return np.alltrue(periods > 0)

def _check_shapes(dim_x, mu, sigma, periods):
    """check the dimensions of inputs
    
    Parameters
    ----------
    dim_x : scalar 
        The dimension of x
    mu 


    """
