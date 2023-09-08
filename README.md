# Asse_Pricing_with_Neural_Networks_Significance_Tests

#####################################################################
#################       Description of folders       #################
#####################################################################
1. code includes:
    build_nn_model.py, initializes and compiles a NN model.
    discretization.py, generates tau_h^(max) samples for X data.
    test_statistics.py, calculates gradients and tstats of variables.
    test_variable_significance.py, identifies significant variables, calculates p-values and significance frequenies.
    reproduce_results.py, produces Table 2, Figure 2, and Figure 3 from data uploaded.

2. data includes:
    test statstics and scaled tau_h^(max) samples for the training set.
    scaled tau_h^(max) = tau_h^(max)*U(C',ϵ_n)^2, see Theorem 3.2.3 in paper.
    download data from https://drive.google.com/drive/folders/1LRgT72VbG0w3EkTcJXfqY33fIMuSmvKX?usp=drive_link
    put folders Pc50 and Pc100 inside the folder data/

3. output includes:
    .csv and figures produced by reproduce_results.py.

#####################################################################
################      Description of key terms       ################
#####################################################################

# simulation setting
N: number of firms
T: number of months
Pc: number of covariates
# discretization setting
m: number of sampled NN models. See Step 1 in discretization approach
sampled_NN: number of hidden layers in the sampled NN model.
tot_iter: total number of iterations. See n_p in paper.

# test statistics
tstat: a value, for a single covariate
tstats: 1D multiple covariates
tstat_MC: 1D, for a single covariate and all MC repetitions
tstats_MC: 2D, for multiple covariates and all MC repetitions

# tau[h^(max)] samples
tau_sample: 1D, or a single covariate
tau_samples: 2D, for multiple covariates
tau_sample_MC: 2D, for a single covariate and all MC repetitions
tau_samples_MC: 3D, for multiple covariates and all MC repetitions