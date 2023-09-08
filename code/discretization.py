
import time
import numpy as np 

def generate_tau_samples(tot_iter,x,m,num_layers,num_hidden_nodes_ls,activation_hidden,var_idx_ls=[]):
    '''
    Parameter
    ---------
    tot_iter: int 
        total number of iterations
    x: ndarray, (num_obs, num_vars)
        observations of variables
    m: int,
        number of sampled NN models
    num_layers: int
        number of hidden layers in the sampled NN model. 
    num_hidden_nodes_ls: list
        number of nodes in each hidden layer in the sampled NN model. 
    activation_hidden: str
        activation function in the sampled nn model. 
    var_idx_ls: list
        indexes of interested variables. 
        If empty, generate tau[h_max] samples for all variables.

    Return
    ------
    tau_samples: ndarray, (tot_iter, num_given_vars) 
        tau[h^(max)] samples for interested or all variables.
    '''
    from test_statistic import comp_gradient
    t0 = time.time()
    tau_samples = []
    for i in range(1,tot_iter+1):
        print('#iteration',i)
        # Step 1 and 2: approximating ζ-cover of F, and sample a multivariate normal variable and find h_max_hat
        h_max_hat = sample_NN_approximate_h_max(x,seed_value_base=i,m=m,num_layers=num_layers,num_hidden_nodes_ls=num_hidden_nodes_ls,activation_hidden=activation_hidden)
        
        # Step 3: Generate one approximate sample of tau(h_max)
        grads = comp_gradient(x,[h_max_hat],var_idx_ls=var_idx_ls) # (num_given_vars,num_obs)
        sqr_grads = [g**2 for g in grads] 
        msqr_grads = np.mean(sqr_grads,axis=1)
        tau_samples.append(msqr_grads)
        time_cost = (time.time()-t0)/60
        print(time_cost/i,'avg. mins per iteration,',time_cost/i*(tot_iter-i),'mins left')
    
    tau_samples = np.array(tau_samples)
    return tau_samples

def sample_NN_approximate_h_max(x,seed_value_base=0,m=500,num_layers=1,num_hidden_nodes_ls=[25],activation_hidden='sigmoid',\
    l1_penalty=0.001,l2_penalty=0,learning_rate=0.001,weight_initializer='glorot_normal'):
    '''
    Parameter
    ---------
    x: ndarray, (num_obs, num_vars)
        observations of variables
    seed_value_base: int
        set seed_value_base to generate seed values
    m: int,
        number of sampled NN models
    num_layers: int
        number of hidden layers
    num_hidden_nodes_ls: list
        number of nodes in each hidden layer
    activation_hidden: str
        activation functions for hidden layers
    l1_penalty: int or float
        penalty for L1 regularizaton
    l2_penalty: int or float
        penalty for L2 regularizaton
    learning_rate: int or float
        learning rate for optimizer
    weight_initializer: str
        weight initializer

    Return
    ------
    h_max_hat: keras sequential
        the model that serves as an approximate arg max of G. See Step 2 in discretization approach
    '''

    from build_nn_model import initialize_compile_nn

    
    # Step 1: Approximating ζ-cover of F
    pred_ls = [] 
    np.random.seed(seed_value_base)
    seed_values = np.random.choice(range(10000), m, replace=False)
    for i in range(0,m,1):
        seed_value = seed_values[i]
        model = initialize_compile_nn(seed_value=seed_value,clear_sess=True,num_layers=num_layers,num_hidden_nodes_ls=num_hidden_nodes_ls,\
                            kernel_initializer=weight_initializer, bias_initializer=weight_initializer,\
                                learning_rate=learning_rate,l1_penalty=l1_penalty,l2_penalty=l2_penalty,\
                                    activation_hidden=activation_hidden,activation_output='linear',compile_needed=False)  
        pred = model.predict(x,verbose=0) # f(x), (num_obs,1)
        pred = np.reshape(pred,(1,-1))
        pred_ls.append(pred[0,:])
        del model
    pred_ls = np.array(pred_ls) # (num_NN,num_obs)
            
    # Step 2: sample a multivariate normal variable and find h_max_hat
    num_obs = np.shape(x)[0]
    cov_mat = np.dot(pred_ls,pred_ls.T)/num_obs
    multivariate_normal_sample = np.random.multivariate_normal(np.zeros(m), cov_mat)
    # arg max of gaussian process
    print('arg max of gaussian process =',end='')
    max_idx = np.argmax(multivariate_normal_sample)
    print(max_idx,'seed_value =',seed_values[max_idx])
    h_max_hat = initialize_compile_nn(seed_value=seed_values[max_idx],clear_sess=True,num_layers=num_layers,num_hidden_nodes_ls=num_hidden_nodes_ls,\
                        kernel_initializer=weight_initializer, bias_initializer=weight_initializer,\
                            learning_rate=learning_rate,l1_penalty=l1_penalty,l2_penalty=l2_penalty,\
                                activation_hidden=activation_hidden,activation_output='linear',compile_needed=False)
    
    return h_max_hat

