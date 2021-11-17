# Generalized Wrapped Gaussian Mixture Model
# author: https://github.com/ludlows
# 2021-Oct

import numpy as np
np.random.seed(17)
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


def _gwgmixture_prob_x_given_component(x, mu_k, sigma_k, periods):
    """probability of X given kth component

    Parameters
    ----------
    x : array-like of shape (n_features,) 
        The input sample vector.
    mu_k : array-like of shape (n_features,)
        The mean vector of the `k`th component.
    sigma_k : array-like of shape (n_features, n_features)
        The covariance matrix of the  k`th` component.
    periods : array-like of shape (n_features,)

    Returns
    -------
    prob_x_given_k : scalar
        Prob(x | z=k) = WN(x; mu_k, sigma_k, periods)
    """
    dim = x.shape[0]
    const = 1.0 / np.sqrt((2*np.pi)**dim*(np.linalg.det(sigma_k)))
    sigma_k_inv = np.linalg.pinv(sigma_k)
    s = 0
    for w in [-1,0,1]:
        diff = x - w * periods - mu_k
        s += np.exp(-0.5 * np.dot(np.dot(diff, sigma_k_inv), diff))
    return const * s


def _gwgmixture_prob_x_and_prob_component_given_x(X, weights, means, covars, periods):
    """function to compute Prob(x_i) and Prob(z_i=k | x_i), z_i = k denotes x_i belongs to `k`th component.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    weights : array-like of shape (n_components, )
        The weights of each component.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The period vector.
    
    Returns
    -------
    prob_X: array-like of shape (n_samples,)
        The probability of each sample.
    prob_component_given_X: array-like of shape (n_components, n_samples)
        The probability Prob(z_i=k | x_i)
    """
    n_samples = X.shape[0]
    n_components = weights.shape[0]
    prob_component_given_X = np.zeros((n_components, n_samples))
    for k in range(n_components):
        alpha = weights[k]
        mu = means[k,:]
        sigma = covars[k,:,:]
        for i in range(n_samples):
            x = X[i,:]
            prob_component_given_X[k,i] = alpha * _gwgmixture_prob_x_given_component(x, mu, sigma, periods)
    
    prob_X = np.sum(prob_component_given_X, axis=0)
    prob_component_given_X = prob_component_given_X / prob_X
    return prob_X, prob_component_given_X



def _gwgmixture_loss(X, weights, means, covars, periods, prob_component_given_X):
    """ function to compute the loss to verify convergence.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    weights : array-like of shape (n_components, )
        The weights of each component.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The period vector.
    prob_component_given_X:  array-like of shape (n_components, n_samples)
        The probability Prob(z_i=k | x_i) 

    Returns
    -------
    loss: scalar
        The loss
    """
    n_components = weights.shape[0]
    n_samples = X.shape[0]
    loss = 0.0
    for k in range(n_components):
        mu = means[k,:]
        alpha = weights[k]
        sigma = covars[k,:,:]
        for i in range(n_samples):
            x = X[i,:]
            loss += prob_component_given_X[k, i] * (np.log(alpha) + np.log(_gwgmixture_prob_x_given_component(x, mu,sigma, periods)))
    return loss


def _gwgmixture_estimate_means(X, means, covars, periods, prob_X):
    """ estimate mean vector for each component

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    weights : array-like of shape (n_components, )
        The weights of each component.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The period vector.
    prob_X : array-like of shape  (n_samples,)
        The probability of each sample.
    
    Returns
    -------
    new_means : array-like of shape (n_components, n_features)
        The new mean vectors
    """
    n_components = means.shape[0]
    n_samples, n_features = X.shape
    prob_X_inv = 1.0 / prob_X
    new_means = np.zeros_like(means)
    exp_term = np.zeros(n_samples)
    exp_diff_term = np.zeros((n_features, n_samples))
    for k in range(n_components):
        mu = means[k, :]
        sigma = covars[k, :, :]
        sigma_inv = np.linalg.pinv(sigma)
        for i in range(n_samples):
            x = X[i,:]
            cache = [0,0,0]
            exp_diff_term[:, i] = 0
            for w_index, w in enumerate([-1,0,1]):
                diff = x - w * periods - mu
                cache[w_index] = np.exp(-0.5 * np.dot(np.dot(diff, sigma_inv), diff))
                exp_diff_term[:, i] += cache[w_index] * (x - w * periods)
            exp_term[i] = np.sum(cache)

        upper = np.dot(exp_diff_term, prob_X_inv)
        lower = np.dot(prob_X_inv, exp_term)
        new_means[k, :] = upper / lower
    return new_means



def _gwgmixture_estimate_covars(X, means, covars, periods, prob_X):
    """ estimate covariance matrix for each component
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.
    weights : array-like of shape (n_components, )
        The weights of each component.
    means : array-like of shape (n_components, n_features)
        The mean vectors
    covars : array-like of shape (n_components, n_features, n_features)
        The covariance matrices.
    periods : array-like of shape (n_features,)
        The period vector.
    prob_X : array-like of shape  (n_samples,)
        The probability of each sample.
    
    Returns
    -------
    new_covars :  array-like of shape (n_components, n_features, n_features)
        The new covariance matrices
    """
    new_covars = np.zeros_like(covars)
    n_components = means.shape[0]
    n_samples, n_features = X.shape
    prob_X_inv = 1.0 / prob_X
    exp_term = np.zeros(n_samples)
    exp_diff_term = np.zeros((n_features, n_features, n_samples))
    for k in range(n_components):
        mu = means[k,:]
        sigma = covars[k,:,:]
        sigma_inv = np.linalg.pinv(sigma)
        
        for i in range(n_samples):
            x = X[i,:]
            cache = [0,0,0]
            exp_diff_term[:,:,i] = 0
            for w_index, w in enumerate([-1,0,1]):
                diff = x - w * periods - mu
                cache[w_index] = np.exp(-0.5 * np.dot(np.dot(diff, sigma_inv), diff))
                t = diff[:,np.newaxis]
                exp_diff_term[:,:,i] += cache[w_index] * np.matmul(t, t.T)
            exp_term[i] = np.sum(cache)

        upper = np.dot(exp_diff_term, prob_X_inv)
        lower = np.dot(prob_X_inv, exp_term)
        new_covars[k, :, :] = upper / lower
    return new_covars
    




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
    n_components : int
        The number of mixture components.
    
    weights_init : array-like of shape (n_components, )
        The user-provided initial weights.
    
    means_init : array-like of shape (n_components, n_features)
        The user-provided initial means.
    
    covars_init : array-like of shape (n_components, n_features, n_features)
        The user-provided initial covariances. 
    
    periods :  array-like of shape (n_features,)
        The periods on all feature dimensions

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    
    max_iter : int, default=100
        The number of EM iterations to perform.

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
    def __init__(self, n_components, weights_init, means_init, covars_init, periods, tol=1e-3, reg_covar=1e-6,  max_iter=100):
        if n_components <= 0:
            raise ValueError("n_components cannot be zero or negative.")
        self.n_components = n_components
        self.weights_ = weights_init
        self.means_ = means_init
        self.covars_ = covars_init
        self.converged_ = False
        self.periods_ = periods
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
        converged_ : bool
            True when convergence was reached in fit(), False otherwise.
        """
        self.converged_ = False
        n_samples = X.shape[0]
        for num in range(self._max_iter):
            prob_X, prob_component_given_X = _gwgmixture_prob_x_and_prob_component_given_x(X, self.weights_, self.means_, self.covars_, self.periods_)
            loss = _gwgmixture_loss(X, self.weights_, self.means_, self.covars_, self.periods_, prob_component_given_X)
            self.weights_ = np.sum(prob_component_given_X, axis=1) / n_samples
            self.means_ = _gwgmixture_estimate_means(X, self.means_, self.covars_, self.periods_, prob_X)
            self.covars_ = _gwgmixture_estimate_covars(X, self.means_, self.covars_, self.periods_, prob_X)
            new_loss = _gwgmixture_loss(X, self.weights_, self.means_, self.covars_, self.periods_, prob_component_given_X)
            if abs(loss - new_loss) < 0.001:
                self.converged_ = True
                break
        return self.converged_
        


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
    
    def prob(self, X):
        """using Generalized Wrapped Gaussian Mixture model get the probability 

        """
        pass
    