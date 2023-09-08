import numpy as np


def comp_test_statistic(x,model_ls,delta=1e-5,var_idx_ls=[]):
    '''
    compute test statistics for different variables

    Parameter
    ---------
    x: ndarray, (num_obs, num_vars)
        observations of variables
    model_ls: list, (num_models,)
        multiple models when using ensemble approach.
        one model when not. 
    delta: float,
        step size in FDM, grad_i = (f(x_i+delta)-f(x_i)) / delta
    var_idx_ls: list, (num_vars,)
        indexes of interested variables. 
        If empty, compute gradients for all variables.
        If not empty, compute gradients for interested variables only.

    Return
    ------
    tstat: list, (num_vars,)
        test statistics of given variables.
    '''
    grads = comp_gradient(x,model_ls,delta=delta,var_idx_ls=var_idx_ls)
    sqr_grads = [grad**2 for grad in grads] 
    tstat = np.mean(sqr_grads,axis=1) 
    return tstat




def comp_gradient(x,model_ls,delta=1e-5,var_idx_ls=[]):
    '''
    compute gradients for different variables using FDM

    Parameter
    ---------
    x: ndarray, (num_obs, num_vars)
        observations of variables
    model_ls: list, (num_models,)
        multiple models when using ensemble approach.
        one model when not. 
    delta: float,
        step size in FDM, grad_i = (f(x_i+delta)-f(x_i)) / delta
    var_idx_ls: list, (num_vars,)
        indexes of interested variables. 
        If empty, compute gradients for all variables.
        If not empty, compute gradients for interested variables only.

    Return
    ------
    grads: list, (num_vars,num_obs)
        gradients of given variables.
    '''
    y_hat_ls = []
    for model in model_ls:
        y_hat = model.predict(x, verbose=0)
        y_hat_ls.append(y_hat)
    y_hat_avg = np.mean(y_hat_ls,axis=0)

    if var_idx_ls==[]:
        num_vars = np.shape(x)[1]
        var_idx_ls = np.arange(0,num_vars)

    grads = [] 
    for var_idx in var_idx_ls:
        x_plus = x.copy()
        x_plus[:,var_idx] = x_plus[:,var_idx] + delta

        y_hat_forward_ls = [] 
        for model in model_ls:
            y_hat_forward = model.predict(x_plus, verbose=0)
            y_hat_forward_ls.append(y_hat_forward)
        y_hat_forward_avg = np.mean(y_hat_forward_ls,axis=0)

        grad = (y_hat_forward_avg - y_hat_avg) / delta 
        grad = np.reshape(grad,(1,-1))[0]
        grads.append(grad) 
    return grads
