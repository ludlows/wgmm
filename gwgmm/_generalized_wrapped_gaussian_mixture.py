# Generalized Wrapped Gaussian Mixture Model
# author: https://github.com/ludlows
# 2021-Oct


import numpy as np
from numpy.core.fromnumeric import prod

def _unwrap(values, periods):
    """unwrap values using corresponding periods
    
    Parameters
    ----------
    values : array-like of shape (n_samples, n_features)
        The proportions of components of each mixture.
    periods : array-like of shape (n_features,)
        The periods of values
    
    Returns
    -------
    unwrapped : array, shape (n_samples, n_features)
    """
    return np.fmod(values + 0.5 * periods, periods) - 0.5 * periods



def _gernalized_wrapped_gaussian_dist_exp(X, mu, sigma, periods):
    """exponential term of probability density function (PDF) in generalized wrapped gaussian distribution
   
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input vector.
    mu : array-like of shape (n_features,)
        The mean vector.
    sigma : array-like of shape (n_features, n_features)
        The covariance matrix.
    periods : array-like of shape (n_features,)

    Returns
    -------
    pdf : array-like of shape (n_samples,)
        value of probabiity density function
    """
    pdf = np.zeros(X.shape[0])
    period_indices = np.expand_dims(np.array([-1, 0, 1]), 1)
    t = period_indices * periods
    sigma_inv = np.linalg.inv(sigma)
    for i in range(X.shape[0]):
        diff = X[i,:] - t - mu
        exp_term = np.exp(-0.5 * np.matmul(np.matmul(diff, sigma_inv), diff.T))
        pdf[i] = np.trace(exp_term)
    return pdf


def _gernalized_wrapped_gaussian_dist_full(X, mu, sigma, periods):
    """probability density function (PDF) in generalized wrapped gaussian distribution
   
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input vector.
    mu : array-like of shape (n_features,)
        The mean vector.
    sigma : array-like of shape (n_features, n_features)
        The covariance matrix.
    periods : array-like of shape (n_features,)

    Returns
    -------
    pdf : scalar, float
    """
    k = X.shape[1]
    exp_term = _gernalized_wrapped_gaussian_dist_exp(X, mu, sigma, periods)
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
    dim_x : scalar, integer 
        The dimension of x
    mu : array-like of shape (dim_x,)
        The mean vector
    sigma : array-like of shape (dim_x,dim_x)
        The covariance matrix.
    periods : array-like of shape (dim_x,)
        The periods of corresponding dimensions in x
    
    Returns
    -------
    valid : scalar, boolean
    """
    if not mu.shape == (dim_x,):
        return False
    if not sigma.shape == (dim_x, dim_x):
        return False
    if not periods.shape == (dim_x,):
        return False
    return True

def _gwgmixture_prob_X_given_component(X, means, covars, periods):
    """get the probability given each component
       Prob(x | z)
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    
    Returns
    -------
    prob : array-like of shape (n_samples, n_components)
        Prob(x | z)
    """
    prob = np.zeros((X.shape[0], means.shape[0]))
    for k in range(means.shape[0]):
        mean = means[k,:]
        covar = covars[k,:,:]
        prob[:, k] = _gernalized_wrapped_gaussian_dist_full(X, mean, covar, periods)
    return prob


def _gwgmixture_joint_prob_X_and_component(X, means, covars, periods, weights):
    """get the joint probability between x and each component
       Prob(x,z) = Prob(z) * Prob(x | z)
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    weights : array-like of shape (n_components, )
        The weights of each component.
    
    Returns
    -------
    prob : array-like of shape (n_samples, n_components)
        Prob(x,z)
    """
    prob_x_given_z = _gwgmixture_prob_X_given_component(X, means, covars, periods)
    return prob_x_given_z * weights

def _gwgmixture_prob_X(X, means, covars, periods, weights):
    """get the marginal probability of x
       Prob(x) = sum_{z} {Prob(z) * Prob(x | z)}
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    weights : array-like of shape (n_components, )
        The weights of each component.
    
    Returns
    -------
    prob : array-like of shape (n_samples, )
        Prob(x)
    """
    return np.sum(_gwgmixture_joint_prob_X_and_component(X, means, covars, periods, weights), axis=1)

def _gwgmixture_loss():
    pass


def _gwgmixture_estimate_means_helper(X, mu, sigma, periods):
    """ helper function to estimate mean vectors of all components
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    mu : array-like of shape (n_features, )
        The mean vectors
    sigma : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    
    Returns
    -------
    periodic_terms : array-like of shape (n_samples, n_features)
        periodic terms weighted by exponential function
    """
    periodic_terms = np.zeros((X.shape[0], mu.shape[0]))
    period_indices = np.expand_dims(np.array([-1, 0, 1]), 1)
    t = period_indices * periods # (3, n_features)
    sigma_inv = np.linalg.inv(sigma)
    for i in range(X.shape[0]):
        term = X[i,:] - t
        diff =  term - mu
        exp_terms = np.exp(-0.5 * np.matmul(np.matmul(diff, sigma_inv), diff.T))
        diag_terms = np.array([exp_terms[dim,dim] for dim in range(len(period_indices))])
        periodic_terms[i] = np.dot(term.T, diag_terms)
    return periodic_terms

def _gwgmixture_estimate_means(X, means, covars, periods, weights):
    """ estimate mean vectors of all components
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    weights : array-like of shape (n_components, )
        The weights of each component.
    
    Returns
    -------
    means_ : array-like of shape (n_compontes, n_features)
         mean vectors for all components
    """
    n_features = X.shape[1]
    n_components = means.shape[0]
    prob_x = _gwgmixture_prob_X(X, means, covars, periods, weights)
    prob_x_inv = 1.0 / prob_x
    means_ = np.zeros((n_components, n_features))
    for k in range(n_components):
        mu = means[k,:]
        sigma = covars[k,:,:]
        exp_terms_down = _gernalized_wrapped_gaussian_dist_exp(X, mu, sigma, periods)
        down = np.dot(exp_terms_down, prob_x_inv)
        exp_terms_up = _gwgmixture_estimate_means_helper(X, mu, sigma, periods)
        up = np.dot(exp_terms_up.T, prob_x_inv)
        means_[k, :] = up / down
    return means


def _gwgmixture_estimate_covars_helper(X, mu, sigma, periods):
    """ helper function to estimate covariance matrices of all components
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    mu : array-like of shape (n_features, )
        The mean vectors
    sigma : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    
    Returns
    -------
    periodic_terms : array-like of shape (n_samples, n_features, n_features)
        periodic terms weighted by exponential function
    """
    n_features = mu.shape[0]
    periodic_terms = np.zeros((X.shape[0], n_features, n_features))
    period_indices = np.expand_dims(np.array([-1, 0, 1]), 1)
    t = period_indices * periods # (3, n_features)
    sigma_inv = np.linalg.inv(sigma)
    temporary_covar = np.zeros((n_features, n_features, len(period_indices)))
    for i in range(X.shape[0]):
        term = X[i,:] - t
        diff =  term - mu
        for k in range(len(period_indices)):
            vector = diff[k, :]
            temporary_covar[:,:,k] = np.matmul(vector[:,np.newaxis], vector[np.newaxis,:])
        exp_terms = np.exp(-0.5 * np.matmul(np.matmul(diff, sigma_inv), diff.T))
        diag_terms = np.array([exp_terms[dim,dim] for dim in range(len(period_indices))])
        periodic_terms[i,:,:] = np.dot(temporary_covar, diag_terms)
    return periodic_terms

def _gwgmixture_estimate_covars(X, means, covars, periods, weights):
    """ estimate covariance matrices of all components
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    weights : array-like of shape (n_components, )
        The weights of each component.
    
    Returns
    -------
    covars_ : array-like of shape (n_compontes, n_features, n_features)
        The covariance matrices of all components
    """
    n_features = X.shape[1]
    n_components = means.shape[0]
    prob_x = _gwgmixture_prob_X(X, means, covars, periods, weights)
    prob_x_inv = 1.0 / prob_x
    covars_ = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        mu = means[k,:]
        sigma = covars[k,:,:]
        exp_terms_down = _gernalized_wrapped_gaussian_dist_exp(X, mu, sigma, periods)
        down = np.dot(exp_terms_down, prob_x_inv)
        exp_terms_up = _gwgmixture_estimate_covars_helper(X, mu, sigma, periods)
        up = np.dot(exp_terms_up.T, prob_x_inv)
        covars_[k, :, :] = up / down
    return covars_

def _gwgmixture_estimate_weights(X, means, covars, periods, weights):
    """estimate weights of all components

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The periods.
    weights : array-like of shape (n_components, )
        The weights of each component.
    
    Returns
    -------
    weights_ : array-like of shape (n_compontes,)
        The covariance matrices of all components
    """
    prob_x_z = _gwgmixture_joint_prob_X_and_component(X, means, covars, periods, weights)
    prob_x = _gwgmixture_prob_X(X, means, covars, periods, weights)
    prob_x_inv = 1.0 / prob_x
    weights_ = np.dot(prob_x_z.T, prob_x_inv) / X.shape[0]
    return weights_




class GWGMixture:
    """Generalized Wrapped Gaussian Mixture Model

    This class allows to estimate the parameters of a Generalized Wrapped Gaussian mixture distribution for angular-value clustering.
    Reference:
        @phdthesis{wang2020speech,
        title={Speech Enhancement using Fiber Acoustic Sensor},
        author={Wang, Miao},
        year={2020},
        school={Concordia University}}
    
    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
   
    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    
    max_iter : int, default=100
        The number of EM iterations to perform.
    
    weights_init : array-like of shape (n_components, )
        The user-provided initial weights.
    
    means_init : array-like of shape (n_components, n_features)
        The user-provided initial means.
    
    covars_init : array-like of shape (n_components, n_features, n_features)
        The user-provided initial covariances. 

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.
    
    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covars_ : array-like of shape (n_components, n_features, n_features)
    
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    """
    def __init__(self, n_components, weights_init, means_init, covars_init, tol=1e-3, reg_covar=1e-6,  max_iter=100):
        if n_components <= 0:
            raise ValueError("n_components cannot be zero or negative.")
        self.n_components = n_components
        self.weights_ = weights_init
        self.means_ = means_init
        self.covars_ = covars_init
        self.converged_ = False
        self.n_iter_ = 0
        self._tol = tol
        self._reg_covar = reg_covar
        self._max_iter = max_iter

    def fit(self, X):
        """using EM algorithm to estimate the parameters of Generalized Wrapped Gaussian Mixture model

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features)
            The input samples.
        
        Returns
        -------
        (weights_, means_,  covar_, converged_)
        """
        pass
        


    def predict(self, X):
        """using Generalized Wrapped Gaussian Mixture model to predict the cluster of each sample in X
        
        Parameters
        ----------
        X : array-like of shape(n_samples, n_features)
            The input samples.
        
        Returns:
        ----------
        y : array-like of shape(n_samples, )
           The predicted cluster index 
        """
        pass