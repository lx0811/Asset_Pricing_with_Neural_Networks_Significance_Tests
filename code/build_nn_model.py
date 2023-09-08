import os
os.environ["OMP_NUM_THREADS"] = '1'
print('os.environ[OMP_NUM_THREADS] =',os.environ["OMP_NUM_THREADS"])

import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import l1,l2,l1_l2
from keras.layers import Dense, BatchNormalization


# %%% define models
def initialize_compile_nn(seed_value=-1, clear_sess=True, num_layers=1, num_hidden_nodes_ls=[32,16,8,4,2],\
                    l1_penalty=0.001, l2_penalty=0, learning_rate=0.001,\
                        kernel_initializer='glorot_uniform', bias_initializer="zeros",\
                            activation_hidden ='relu', activation_output='linear',\
                                compile_needed=True,loss='mse',optimizer='adam'):
    '''
    initialize a NN model and compile
    
    Parameters
    ----------
    seed_value: int
        set seed for replicating models.
        if seed value is -1, randomly initialize a model.
        if not -1, initialize a model based on the given seed value.
    clear_sess: bool
        if True, clear global state. 
        Reset the previous state when initializing a new model.
    num_layers: int
        number of hidden layers
    num_hidden_nodes_ls: list
        number of nodes in each hidden layer
    l1_penalty: int or float
        penalty for L1 regularizaton
    l2_penalty: int or float
        penalty for L2 regularizaton
    learning_rate: int or float
        learning rate for optimizer
    kernel_initializer: str
        weight initializer
    bias_initializer: str
        bias initializer
    activation_hidden: str
        activation functions for hidden layers
    activation_output: str
        activation functions for output layer
    compile_needed: bool
        if True, compile the initialized model.
    loss: str
        loss function to minimize when compiling model
    optimizer: str
        optimizer when compiling model
    
    Returns
    -------
    model: keras sequential
        a compiled or initialized NN model
    '''

    # clear global state
    if clear_sess==True:
        # print('clear session')
        tf.compat.v1.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
    
    # to replicate a NN model
    if seed_value!=-1:
        print('seed_value =',seed_value)
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value) 
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,\
        device_count={'CPU':1})
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)


    model = Sequential()
    for i in range(num_layers):
        if len(num_hidden_nodes_ls)==1:
            num_nodes = int(num_hidden_nodes_ls[0])
        else:
            num_nodes = int(num_hidden_nodes_ls[i])
        model.add(Dense(num_nodes,kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,\
            kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
                bias_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
                    activity_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
                        activation=activation_hidden))
        model.add(BatchNormalization())   
    model.add(Dense(1,kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,\
        kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
            bias_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
                activity_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty),\
                    activation=activation_output))
    if compile_needed==True:
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        if optimizer == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        if optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model