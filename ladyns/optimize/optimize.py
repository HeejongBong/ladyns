import time

import numpy as np
from numpy import linalg

import ladyns.optimize.core as core

def fit(observations, lambda_glasso, lambda_ridge=0, 
        ths = 0.001, max_iter = 1000, 
        ths_glasso = 1e-8, ths_lasso = 1e-10, verbose = False,
        weight_init = None, Omega_init = None):
    
    # check on observation
    assert(observations[0].ndim == 2)
        
    # check on lambda_ridge
    if isinstance(lambda_ridge, (int, float)):
        lambda_ridge = np.full(len(observations), lambda_ridge)
    else:
        assert(len(lambda_ridge) == len(observations))  
    
    # check on lambda_glasso                   
    assert(lambda_glasso.shape == (len(observations),len(observations)))
    
    # initialization
    if weight_init is None:
        weight = [np.ones(np.shape(obs)[0]) for obs in observations]
    else:
        weight = weight_init.copy()
    for p, (w, obs) in enumerate(zip(weight, observations)):
        weight[p] = w / np.sqrt(np.var(w @ obs) + lambda_ridge[p] * w @ w)
    latent = np.array([w @ obs for obs, w in zip(observations, weight)])
    
    Sigma_pre = np.cov(latent, bias=True) * (1 - np.eye(len(observations))) \
        + np.eye(len(observations))    
    if Omega_init is None:
        Sigma = Sigma_pre + lambda_glasso * np.eye(len(observations))
        Omega = linalg.inv(Sigma)
    else:
        Omega = Omega_init.copy()
        Sigma = linalg.inv(Omega)
    
    # iteration
    for iter_miccs in range(max_iter):
        Sigma_last_miccs = Sigma.copy()
        start_iter = time.time()

        # glasso procedure
        core.glasso(Omega, Sigma, Sigma_pre, lambda_glasso, 
                    ths_glasso, max_iter, ths_lasso, max_iter)

        # update weight and latent
        for p in range(Sigma.shape[0]):
            n_p = len(observations[p])
            Dp = np.logical_and(lambda_glasso[p] >= 0, np.arange(Sigma.shape[0])!=p)
            Sp = np.cov(observations[p], latent[Dp], bias=True)
            wp = np.linalg.inv(Sp[:n_p,:n_p] + lambda_ridge[p]*np.eye(n_p)) \
                 @ Sp[:n_p,n_p:] @ Omega[p,Dp]
            
            offset = np.sqrt(np.var(wp @ observations[p]) + lambda_ridge[p] * wp @ wp)
            if offset != 0:
                weight[p] = - wp / offset
            latent[p] = weight[p] @ observations[p]

        # calculate correlation
        Sigma_pre = (np.cov(latent, bias=True) 
                     * (1 - np.eye(len(observations))) 
                     + np.eye(len(observations)))

        nll = np.linalg.slogdet(Sigma)[1] + np.sum(Omega * Sigma_pre) \
              + np.sum(lambda_glasso * np.abs(Omega))
        lapse = time.time() - start_iter
        change = np.max(np.abs(Sigma - Sigma_last_miccs))
        if verbose:
            print("%d-th iter, nll: %f, change: %f, lapse: %f"
                  %(iter_miccs+1, nll, change, lapse))
        if(change < ths):
            break
            
    return Omega, Sigma_pre, latent, weight

def nll(observations, Omega_est, weight_est,
        lambda_ridge=0, mean_train=None, cov_train=None):
    assert(len(observations) == Omega_est.shape[0])
           
    logdet_Omega = np.linalg.slogdet(Omega_est)
    assert(logdet_Omega[0] > 0)
    
    if mean_train is None:
        mean_train = [np.mean(obs_i, -1, keepdims=True) for obs_i in observations]
    if cov_train is None:
        cov_train = [np.cov(obs_i, bias=True) + lambda_ridge * np.eye(obs_i.shape[0])
                     for obs_i in observations]
        
    dev_test = [obs_i - m_i for obs_i, m_i in zip(observations, mean_train)]
    sdm_test = [dev_i_test @ dev_i_test.T / dev_i_test.shape[1]
                for dev_i_test in dev_test]

    nll = (- logdet_Omega[1]
           + np.sum([np.sum(np.linalg.inv(cov_i_train) * sdm_i_test)
                     for cov_i_train, sdm_i_test 
                     in zip(cov_train, sdm_test)])
           + np.sum((Omega_est - np.eye(Omega_est.shape[0]))
                    * np.cov([weight_i_est @ dev_i_test for weight_i_est, dev_i_test
                              in zip(weight_est, dev_test)], bias=True)))
    return nll

def bic(observations, Omega_est, weight_est,
        lambda_ridge=0, mean_train=None, cov_train=None):
    
    num_trial = observations[0].shape[-1]
    nll_est = nll(observations, Omega_est, weight_est, 
                  lambda_ridge, mean_train, cov_train)
    
    return (2 * nll_est * num_trial + np.log(num_trial) * np.sum(Omega_est != 0))

def imshow(image, vmin=None, vmax=None, cmap='RdBu', **kwargs):
    image = np.array(image).astype(float)
    assert(image.ndim == 2)

    # get vmin, vmax
    if vmax is None:
        vmax = np.maximum(np.max(np.abs(image)), 1e-6)
    if vmin is None:
        vmin = -vmax
    
    # get figure    
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)