
import numpy as np

def comp_pvalue(scaled_tau_sample,tstat):
    '''
    Compute p-value for a variable

    Parameter
    ---------
    scaled_tau_sample: list, (sample_len,)
        scaled tau[h^(max)] sample for a variable
    tstat: float,
        test statistic of this variable

    Return
    ------
    pval: float,
        p-value for this variable.
    '''

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(scaled_tau_sample) # empirical cdf of tau samples
    pval = 1 - ecdf(tstat)
    return pval
    
def find_significant_variables(tstats,scaled_tau_samples,alpha,var_names=[]):
    '''
    find significant variables at given significance level

    Parameter
    ---------
    scaled_tau_samples: ndarray, (sample_len,num_vars)
        scaled tau[h^(max)] samples for multiple variables
    tstats: list, (num_vars,)
        test statistics of multiple variables
    alpha: float,
        significance level
    var_names: list, (num_vars,)
        variables names

    Return
    ------
    sign_vars: list,
        indexes or names of significant variables
    '''
    quantile = np.quantile(scaled_tau_samples,q=1-alpha,axis=0,method='linear') 
        
    sign_vars = np.where(tstats>=quantile)[0] # only significant variables
    if var_names!=[]:
        sign_vars = [var_names[v] for v in sign_vars]
    return sign_vars
    

def variable_significance_frequency(tstats_MC,scaled_tau_samples_MC,alpha,var_names=[]):
    '''
    the frequency that a variable is identified as significant.
    frequency: Out of all Monte Carlo repetitions, how many times the variable is identified as significant

    Parameter
    ---------
    scaled_tau_samples_MC: ndarray, (num_MC,sample_len,num_vars)
        scaled tau[h^(max)] samples for multiple variables for all Monte Carlo repetitions
    tstats_MC: ndarray, (num_MC,num_vars)
        test statistics of multiple variables for all Monte Carlo repetitions
    alpha: float,
        significance level
    var_names: list, (num_vars,)
        variables names

    Return
    ------
    var_sign_freq_dict: dictionary, 
        keys are significant variables and values are their frequencies, sorted by values.
    '''
    sign_vars_MC = []
    num_MC = 0
    for tstats,scaled_tau_samples in zip(tstats_MC,scaled_tau_samples_MC): 
        sign_vars = find_significant_variables(tstats,scaled_tau_samples,alpha,var_names)
        sign_vars_MC.append(sign_vars)
        num_MC += 1

    import itertools as it
    from collections import Counter
    var_sign_freq_dict = dict()
    var_sign_freq_dict = Counter(it.chain(*map(set, sign_vars_MC)))
    for item, count in var_sign_freq_dict.items():
        var_sign_freq_dict[item] = round(count/num_MC*100,1)
    var_sign_freq_dict = dict(sorted(var_sign_freq_dict.items(), key=lambda item: item[1], reverse=True))
    
    return var_sign_freq_dict
