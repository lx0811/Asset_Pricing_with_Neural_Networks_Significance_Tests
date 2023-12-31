# Asset_Pricing_with_Neural_Networks_Significance_Tests

This repository contains code, data, and results of the paper **[Asset pricing with neural networks: Significance tests](https://www.sciencedirect.com/science/article/pii/S0304407623002907)**, co-authored by [Hasan Fallahgoul](https://www.hfallahgoul.com/), Vincentius Franstianto and [Xin Lin](https://sites.google.com/view/xin-lin)

## Version

Python 3.8.10

Tensorflow 2.8.0

Keras 2.8.0

Numpy 1.24.1

Pandas 1.3.3

matplotlib 3.4.3

## Download data

download data from https://drive.google.com/drive/folders/1LRgT72VbG0w3EkTcJXfqY33fIMuSmvKX?usp=drive_link

put Pc50 and Pc100 inside the folder data

## Description of folders     

1. **code** includes:

    **build_nn_model.py**, initializes and compiles a NN model.

    **discretization.py**, generates tau_h^(max) samples for X data.

    **test_statistics.py**, calculates gradients and tstats of variables.

    **test_variable_significance.py**, identifies significant variables, calculates p-values and significance frequenies.

    **reproduce_results.py**, produces Table 2, Figure 2, and Figure 3 from data uploaded.

3. **data** includes:

    **test statistics** and scaled **tau_h^(max)** samples for the training set.

    scaled tau_h^(max) = tau_h^(max)*U(C',ϵ_n)^2, see Theorem 3.2.3 in paper.

4. **output** includes:

    .csv and figures produced by **reproduce_results.py**.
