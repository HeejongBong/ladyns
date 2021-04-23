import numpy as np
import ladyns.optimize as mx

def _generate_lambda_glasso(bin_num, lambda_glasso, offset, lambda_diag=None):
    lambda_glasso_out = np.full((bin_num, bin_num), -1) + (1+lambda_glasso) * \
           (np.abs(np.arange(bin_num) - np.arange(bin_num)[:,np.newaxis]) <= offset)
    if lambda_diag:
        lambda_glasso_out[np.arange(bin_num), np.arange(bin_num)] = lambda_diag
    return lambda_glasso_out

def fit(populations, lambda_diag, lambda_cross, offset_cross, 
        lambda_auto=None, offset_auto=None, lambda_ridge=0, 
        adj_sign=True, weight_org=None, **kwargs):
    
    num_time = populations[0].shape[0]
    
    observations = list()
    for pop in populations:
        observations = observations + list(pop)
    
    # get full_graph
    if lambda_auto is None:
        lambda_auto = lambda_cross
    if offset_auto is None:
        offset_auto = offset_cross
    lambda_glasso_auto = _generate_lambda_glasso(num_time, lambda_auto, 
                                                 offset_auto, lambda_diag=lambda_diag)
    lambda_glasso_cross = _generate_lambda_glasso(num_time, lambda_cross,
                                                  offset_cross)
    lambda_glasso = np.array(np.block(
        [[lambda_glasso_auto if j==i else lambda_glasso_cross
          for j, _ in enumerate(populations)]
         for i, _ in enumerate(populations)]))
        
    # run
    Omega, Sigma, latent, weight \
    = mx.fit(observations, lambda_glasso, lambda_ridge, **kwargs)
    
    # adjust sign
    if adj_sign:
        if weight_org is None:
            Omega, Sigma, latent, weight \
            = adj_sign_est(Omega, Sigma, latent, weight, num_time)
        else:
            Omega, Sigma, latent, weight \
            = adj_sign_bst(Omega, Sigma, latent, weight, weight_org)

    return Omega, Sigma, latent, weight

def nll(populations, Omega_est, weight_est,
        lambda_ridge=0, means_pop=None, covs_pop=None):
    
    observations = list()
    for pop in populations:
        observations = observations + list(pop)
        
    if means_pop is None:
        mean_train = [np.mean(obs_i, -1, keepdims=True) for obs_i in observations]
    else:
        mean_train = list()
        for mean_pop in means_pop:
            mean_train = mean_train + list(mean_pop)
            
    if covs_pop is None:
        cov_train = [np.cov(obs_i, bias=True) + lambda_ridge * np.eye(obs_i.shape[0])
                     for obs_i in observations]
    else:
        cov_train = list()
        for cov_pop in covs_pop:
            cov_train = cov_train + list(cov_pop)
            
    return mx.nll(observations, Omega_est, weight_est, lambda_ridge,
                  mean_train, cov_train)

def bic(populations, Omega_est, weight_est,
        lambda_ridge=0, means_pop=None, covs_pop=None):
    
    num_trial = populations[0].shape[-1]
    nll_est = nll(populations, Omega_est, weight_est, 
                  lambda_ridge, means_pop, covs_pop) 
    
    return (2 * nll_est * num_trial + np.log(num_trial) * np.sum(Omega_est != 0))
    
def adj_sign_est(precision, correlation, latent, weight, num_time):
    assert(precision.shape[0] % num_time == 0)
    num_pop = int(precision.shape[0] / num_time)
    
    temp_sign = np.cumprod(np.concatenate([
        np.concatenate([
            np.ones((1)),
            np.sign(np.sign(correlation[np.arange(i*num_time,(i+1)*num_time-1),
                                        np.arange(i*num_time+1,(i+1)*num_time)])+0.5)])
        for i in range(num_pop)]))

    correlation = correlation *\
        temp_sign.reshape((2*num_time, 1)) * temp_sign.reshape((1, 2*num_time))
    precision = precision *\
        temp_sign.reshape((2*num_time, 1)) * temp_sign.reshape((1, 2*num_time))
    latent = latent * temp_sign.reshape((2*num_time, 1))
    weight = [w*sgn for w, sgn in zip(weight, temp_sign)]

    temp_sign = np.cumprod(np.concatenate([
        np.ones((num_time)),
        np.concatenate([
            np.concatenate([
                np.sign(np.sign(np.sum(np.sign(
                    correlation[np.arange((i-1)*num_time,i*num_time-1), 
                                np.arange(i*num_time, (i+1)*num_time-1)]
                )))+0.5).reshape((1)),
                np.ones((num_time-1))])
            for i in range(1, num_pop)])
    ]))

    correlation = correlation *\
        temp_sign.reshape((2*num_time, 1)) * temp_sign.reshape((1, 2*num_time))
    precision = precision *\
        temp_sign.reshape((2*num_time, 1)) * temp_sign.reshape((1, 2*num_time))
    latent = latent * temp_sign.reshape((2*num_time, 1))
    weight = [w*sgn for w, sgn in zip(weight, temp_sign)]
    
    return precision, correlation, latent, weight
    
def adj_sign_bst(prec_bst, corr_bst, latent_bst, weights_bst, weights_est):    
    temp_sign = np.array([np.sign(np.sum(w_bst * w_est)) for w_bst, w_est
                          in zip(weights_bst, weights_est)])
    
    prec_bst = prec_bst * temp_sign * temp_sign[:,None]
    corr_bst = corr_bst * temp_sign * temp_sign[:,None]
    latent_bst = latent_bst * temp_sign[:,None]
    weights_bst = [w*sgn for w, sgn in zip(weights_bst, temp_sign)]
    
    return prec_bst, corr_bst, latent_bst, weights_bst
    
def lead_lag(Omegas_cross):
    # assert on shape of Omegas_cross
    assert(Omegas_cross.shape[-1] == Omegas_cross.shape[-2])
    num_time = Omegas_cross.shape[-1]
    
    lead_lags = np.zeros(Omegas_cross.shape[:-2]+(2*num_time-1,))
    for t in range(2*num_time-1):
        ind_times = np.arange(max(0, t-num_time+1), min(num_time, t+1))
        Omegas_t = Omegas_cross[..., ind_times, ind_times[::-1]]

        lead_lags[...,t] = (
            (np.abs(Omegas_t) @ (ind_times[::-1]-ind_times))
            /np.where(np.sum(np.abs(Omegas_t), -1) > 0,
                      np.sum(np.abs(Omegas_t), -1), np.nan))
    return lead_lags

def imshow(image, vmin=None, vmax=None, cmap='RdBu', time=None, identity=False, **kwargs):
    if time:
        assert(image.shape[0] == image.shape[1])
        kwargs['extent'] = [time[0], time[1], time[1], time[0]]
    
    # get figure   
    mx.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)   
        
    if identity and time:
        import matplotlib.pyplot as plt
        plt.plot([time[0], time[1]], [time[0], time[1]], linewidth = 0.3, color='black')