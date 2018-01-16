import networkx as nx
import numpy as np


def getSPLengths(G1):
    sp = nx.shortest_path(G1)
    distances = np.zeros((G1.number_of_nodes(), G1.number_of_nodes()))
    for i in sp.keys():
        for j in sp[i].keys():
            distances[i, j] = len(sp[i][j])-1
    return distances

def getSPGraph(G, edge_weight = 'bond_type'):
    """Transform graph G to its corresponding shortest-paths graph.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.
        
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
        
    Notes
    ------
    For an input graph G, its corresponding shortest-paths graph S contains the same set of nodes as G, while there exists an edge between all nodes in S which are connected by a walk in G. Every edge in S between two nodes is labeled by the shortest distance between these two nodes.
    
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    return floydTransformation(G, edge_weight = edge_weight)
            
def floydTransformation(G, edge_weight = 'bond_type'):
    """Transform graph G to its corresponding shortest-paths graph using Floyd-transformation.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.
        
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
        
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    spMatrix = nx.floyd_warshall_numpy(G, weight = edge_weight)
    S = nx.Graph()
    S.add_nodes_from(G.nodes(data=True))
    for i in range(0, G.number_of_nodes()):
        for j in range(0, G.number_of_nodes()):
            S.add_edge(i, j, cost = spMatrix[i, j])
    return S



import os
import pathlib
from collections import OrderedDict
from tabulate import tabulate
from .graphfiles import loadDataset

def kernel_train_test(datafile, kernel_file_path, kernel_func, kernel_para, trials = 100, splits = 10, alpha_grid = None, C_grid = None, hyper_name = '', hyper_range = [1], normalize = False):
    """Perform training and testing for a kernel method. Print out neccessary data during the process then finally the results.
    
    Parameters
    ----------
    datafile : string
        Path of dataset file.
    kernel_file_path : string
        Path of the directory to save results.
    kernel_func : function
        kernel function to use in the process.
    kernel_para : dictionary
        Keyword arguments passed to kernel_func.
    trials: integer
        Number of trials for hyperparameter random search, where hyperparameter stands for penalty parameter for now. The default is 100.
    splits: integer
        Number of splits of dataset. Times of training and testing procedure processed. The final means and stds are the average of the results of all the splits. The default is 10.
    alpha_grid : ndarray
        Penalty parameter in kernel ridge regression. Corresponds to (2*C)^-1 in other linear models such as LogisticRegression.
    C_grid : ndarray
        Penalty parameter C of the error term in kernel SVM.
    hyper_name : string
        Name of the hyperparameter.
    hyper_range : list
        Range of the hyperparameter.
    normalize : string
        Determine whether or not that normalization is performed. The default is False.

    References
    ----------
    [1] Elisabetta Ghisu, https://github.com/eghisu/GraphKernels/blob/master/GraphKernelsCollection/python_scripts/compute_perf_gk.py, 2018.1
    
    Examples
    --------
    >>> import sys
    >>> sys.path.insert(0, "../")
    >>> from pygraph.utils.utils import kernel_train_test
    >>> from pygraph.kernels.treeletKernel import treeletkernel
    >>> datafile = '../../../../datasets/acyclic/Acyclic/dataset_bps.ds'
    >>> kernel_file_path = 'kernelmatrices_path_acyclic/'
    >>> kernel_para = dict(node_label = 'atom', edge_label = 'bond_type', labeled = True)
    >>> kernel_train_test(datafile, kernel_file_path, treeletkernel, kernel_para, normalize = True)
    """
    # setup the parameters
    model_type = 'regression' # Regression or classification problem
    print('\n --- This is a %s problem ---' % model_type)
    
    alpha_grid = np.logspace(-10, 10, num = trials, base = 10) if alpha_grid == None else alpha_grid # corresponds to (2*C)^-1 in other linear models such as LogisticRegression
    C_grid = np.logspace(-10, 10, num = trials, base = 10) if C_grid == None else C_grid
    
    if not os.path.exists(kernel_file_path):
        os.makedirs(kernel_file_path)
        
    train_means_list = []
    train_stds_list = []
    test_means_list = []
    test_stds_list = []
    kernel_time_list = []
        
    for hyper_para in hyper_range:
        print('' if hyper_name == '' else '\n\n #--- calculating kernel matrix when %s = %.1f ---#' % (hyper_name, hyper_para))

        print('\n Loading dataset from file...')
        dataset, y = loadDataset(datafile)
        y = np.array(y)
#             print(y)

        # save kernel matrices to files / read kernel matrices from files
        kernel_file = kernel_file_path + 'km.ds'
        path = pathlib.Path(kernel_file)
        # get train set kernel matrix
        if path.is_file():
            print('\n Loading the kernel matrix from file...')
            Kmatrix = np.loadtxt(kernel_file)
            print(Kmatrix)
        else:
            print('\n Calculating kernel matrix, this could take a while...')
            if hyper_name != '':
                kernel_para[hyper_name] = hyper_para
            Kmatrix, run_time = kernel_func(dataset, **kernel_para)
            kernel_time_list.append(run_time)
            print(Kmatrix)
            print('\n Saving kernel matrix to file...')
        #     np.savetxt(kernel_file, Kmatrix)

        """
        -  Here starts the main program
        -  First we permute the data, then for each split we evaluate corresponding performances
        -  In the end, the performances are averaged over the test sets
        """

        train_mean, train_std, test_mean, test_std = \
            split_train_test(Kmatrix, y, alpha_grid, C_grid, splits, trials, model_type, normalize = normalize)

        train_means_list.append(train_mean)
        train_stds_list.append(train_std)
        test_means_list.append(test_mean)
        test_stds_list.append(test_std)

    print('\n')
    table_dict = {'RMSE_test': test_means_list, 'std_test': test_stds_list, \
        'RMSE_train': train_means_list, 'std_train': train_stds_list, 'k_time': kernel_time_list}
    if hyper_name == '':
        keyorder = ['RMSE_test', 'std_test', 'RMSE_train', 'std_train', 'k_time']

    else:
        table_dict[hyper_name] = hyper_range
        keyorder = [hyper_name, 'rmse_test', 'std_test', 'rmse_train', 'std_train', 'k_time']
    print(tabulate(OrderedDict(sorted(table_dict.items(), key = lambda i:keyorder.index(i[0]))), headers='keys'))


import random
from sklearn.kernel_ridge import KernelRidge # 0.17
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import svm

def split_train_test(Kmatrix, train_target, alpha_grid, C_grid, splits = 10, trials = 100, model_type = 'regression', normalize = False):
    """Split dataset to training and testing splits, train and test. Print out and return the results.
    
    Parameters
    ----------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the kernel between 2 praphs.
    train_target : ndarray
        train target.
    alpha_grid : ndarray
        Penalty parameter in kernel ridge regression. Corresponds to (2*C)^-1 in other linear models such as LogisticRegression.
    C_grid : ndarray
        Penalty parameter C of the error term in kernel SVM.
    splits : interger
        Number of splits of dataset. Times of training and testing procedure processed. The final means and stds are the average of the results of all the splits. The default is 10.
    trials : integer
        Number of trials for hyperparameters random search. The final means and stds are the ones in the same trial with the best test mean. The default is 100.
    model_type : string
        Determine whether it is a regression or classification problem. The default is 'regression'.
    normalize : string
        Determine whether or not that normalization is performed. The default is False.
        
    Return
    ------
    train_mean : float
        mean of train accuracies in the same trial with the best test mean.
    train_std : float
        mean of train stds in the same trial with the best test mean.
    test_mean : float
        mean of the best tests.
    test_std : float
        mean of test stds in the same trial with the best test mean.
    
    References
    ----------
    [1] Elisabetta Ghisu, https://github.com/eghisu/GraphKernels/blob/master/GraphKernelsCollection/python_scripts/compute_perf_gk.py, 2018.1
    """
    datasize = len(train_target)
    random.seed(20) # Set the seed for uniform parameter distribution
    
    # Initialize the performance of the best parameter trial on train with the corresponding performance on test
    train_split = []
    test_split = []

    # For each split of the data
    for j in range(10, 10 + splits):
    #         print('\n Starting split %d...' % j)

        # Set the random set for data permutation
        random_state = int(j)
        np.random.seed(random_state)
        idx_perm = np.random.permutation(datasize)

        # Permute the data
        y_perm = train_target[idx_perm] # targets permutation
        Kmatrix_perm = Kmatrix[:, idx_perm] # inputs permutation
        Kmatrix_perm = Kmatrix_perm[idx_perm, :] # inputs permutation

        # Set the training, test
        # Note: the percentage can be set up by the user
        num_train = int((datasize * 90) / 100)         # 90% (of entire dataset) for training
        num_test = datasize - num_train             # 10% (of entire dataset) for test

        # Split the kernel matrix
        Kmatrix_train = Kmatrix_perm[0:num_train, 0:num_train]
        Kmatrix_test = Kmatrix_perm[num_train:datasize, 0:num_train]

        # Split the targets
        y_train = y_perm[0:num_train]
       

        # Normalization step (for real valued targets only)
        if normalize == True and model_type == 'regression':
            y_train_mean = np.mean(y_train)
            y_train_std = np.std(y_train)
            y_train_norm = (y_train - y_train_mean) / float(y_train_std)

        y_test = y_perm[num_train:datasize]

        # Record the performance for each parameter trial respectively on train and test set
        perf_all_train = []
        perf_all_test = []

        # For each parameter trial
        for i in range(trials):
            # For regression use the Kernel Ridge method
            if model_type == 'regression':
                # Fit the kernel ridge model
                KR = KernelRidge(kernel = 'precomputed', alpha = alpha_grid[i])
    #                 KR = svm.SVR(kernel = 'precomputed', C = C_grid[i])
                KR.fit(Kmatrix_train, y_train if normalize == False else y_train_norm)

                # predict on the train and test set
                y_pred_train = KR.predict(Kmatrix_train)
                y_pred_test = KR.predict(Kmatrix_test)

                # adjust prediction: needed because the training targets have been normalized
                if normalize == True:
                    y_pred_train = y_pred_train * float(y_train_std) + y_train_mean                
                    y_pred_test = y_pred_test * float(y_train_std) + y_train_mean

                # root mean squared error in train set
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                perf_all_train.append(rmse_train)
                # root mean squared error in test set
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                perf_all_test.append(rmse_test)

        # --- FIND THE OPTIMAL PARAMETERS --- #
        # For regression: minimise the mean squared error
        if model_type == 'regression':

            # get optimal parameter on test (argmin mean squared error)
            min_idx = np.argmin(perf_all_test)
            alpha_opt = alpha_grid[min_idx]

            # corresponding performance on train and test set for the same parameter
            perf_train_opt = perf_all_train[min_idx]
            perf_test_opt = perf_all_test[min_idx]

        # append the correponding performance on the train and test set
        train_split.append(perf_train_opt)
        test_split.append(perf_test_opt)

    # average the results
    # mean of the train and test performances over the splits
    train_mean = np.mean(np.asarray(train_split))
    test_mean = np.mean(np.asarray(test_split))
    # std deviation of the train and test over the splits
    train_std = np.std(np.asarray(train_split))
    test_std = np.std(np.asarray(test_split))

    print('\n Mean performance on train set: %3f' % train_mean)
    print('With standard deviation: %3f' % train_std)
    print('\n Mean performance on test set: %3f' % test_mean)
    print('With standard deviation: %3f' % test_std)
    
    return train_mean, train_std, test_mean, test_std