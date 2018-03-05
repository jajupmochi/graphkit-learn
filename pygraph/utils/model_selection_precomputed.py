

def model_selection_for_precomputed_kernel(datafile, estimator,
                                           param_grid_precomputed, param_grid,
                                           model_type, NUM_TRIALS=30,
                                           datafile_y=''):
    """Perform model selection, fitting and testing for precomputed kernels using nested cv. Print out neccessary data during the process then finally the results.

    Parameters
    ----------
    datafile : string
        Path of dataset file.
    estimator : function
        kernel function used to estimate. This function needs to return a gram matrix.
    param_grid_precomputed : dictionary
        Dictionary with names (string) of parameters used to calculate gram matrices as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings.
    param_grid : dictionary
        Dictionary with names (string) of parameters used as penelties as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings.
    model_type : string
        Typr of the problem, can be regression or classification.
    NUM_TRIALS : integer
        Number of random trials of outer cv loop. The default is 30.
    datafile_y : string
        Path of file storing y data. This parameter is optional depending on the given dataset file.

    Examples
    --------
    >>> import numpy as np
    >>> import sys
    >>> sys.path.insert(0, "../")
    >>> from pygraph.utils.model_selection_precomputed import model_selection_for_precomputed_kernel
    >>> from pygraph.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
    >>>
    >>> datafile = '../../../../datasets/acyclic/Acyclic/dataset_bps.ds'
    >>> estimator = weisfeilerlehmankernel
    >>> param_grid_precomputed = {'height': [0,1,2,3,4,5,6,7,8,9,10], 'base_kernel': ['subtree']}
    >>> param_grid = {"alpha": np.logspace(-2, 2, num = 10, base = 10)}
    >>>
    >>> model_selection_for_precomputed_kernel(datafile, estimator, param_grid_precomputed, param_grid, 'regression')
    """
    import numpy as np
    from matplotlib import pyplot as plt

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split, ParameterGrid

    import sys
    sys.path.insert(0, "../")
    from pygraph.utils.graphfiles import loadDataset

    from tqdm import tqdm

    # setup the model type
    model_type = model_type.lower()
    if model_type != 'regression' and model_type != 'classification':
        raise Exception(
            'The model type is incorrect! Please choose from regression or classification.')
    print()
    print('--- This is a %s problem ---' % model_type)

    # Load the dataset
    print()
    print('1. Loading dataset from file...')
    dataset, y = loadDataset(datafile, filename_y=datafile_y)

    # Grid of parameters with a discrete number of values for each.
    param_list_precomputed = list(ParameterGrid(param_grid_precomputed))
    param_list = list(ParameterGrid(param_grid))

    # Arrays to store scores
    train_pref = np.zeros(
        (NUM_TRIALS, len(param_list_precomputed), len(param_list)))
    val_pref = np.zeros(
        (NUM_TRIALS, len(param_list_precomputed), len(param_list)))
    test_pref = np.zeros(
        (NUM_TRIALS, len(param_list_precomputed), len(param_list)))

    gram_matrices = []  # a list to store gram matrices for all param_grid_precomputed
    gram_matrix_time = []  # a list to store time to calculate gram matrices

    # calculate all gram matrices
    print()
    print('2. Calculating gram matrices. This could take a while...')
    for params_out in param_list_precomputed:
        Kmatrix, current_run_time = estimator(dataset, **params_out)
        print()
        print('gram matrix with parameters', params_out, 'is: ')
        print(Kmatrix)
        plt.matshow(Kmatrix)
        plt.colorbar()
        plt.show()
#         plt.savefig('../../notebooks/gram_matrix_figs/{}_{}'.format(estimator.__name__, params_out))
        gram_matrices.append(Kmatrix)
        gram_matrix_time.append(current_run_time)

    print()
    print('3. Fitting and predicting using nested cross validation. This could really take a while...')
    # Loop for each trial
    pbar = tqdm(total=NUM_TRIALS * len(param_list_precomputed) * len(param_list),
                desc='calculate performance', file=sys.stdout)
    for trial in range(NUM_TRIALS):  # Test set level
        # loop for each outer param tuple
        for index_out, params_out in enumerate(param_list_precomputed):
            # split gram matrix and y to app and test sets.
            X_app, X_test, y_app, y_test = train_test_split(
                gram_matrices[index_out], y, test_size=0.1)
            split_index_app = [y.index(y_i) for y_i in y_app if y_i in y]
            split_index_test = [y.index(y_i) for y_i in y_test if y_i in y]
            X_app = X_app[:, split_index_app]
            X_test = X_test[:, split_index_app]
            y_app = np.array(y_app)
            y_test = np.array(y_test)

            # loop for each inner param tuple
            for index_in, params_in in enumerate(param_list):
                inner_cv = KFold(n_splits=10, shuffle=True, random_state=trial)
                current_train_perf = []
                current_valid_perf = []
                current_test_perf = []

                # For regression use the Kernel Ridge method
                if model_type == 'regression':
                    KR = KernelRidge(kernel='precomputed', **params_in)
                    # loop for each split on validation set level
                    # validation set level
                    for train_index, valid_index in inner_cv.split(X_app):
                        KR.fit(X_app[train_index, :]
                               [:, train_index], y_app[train_index])

                        # predict on the train, validation and test set
                        y_pred_train = KR.predict(
                            X_app[train_index, :][:, train_index])
                        y_pred_valid = KR.predict(
                            X_app[valid_index, :][:, train_index])
                        y_pred_test = KR.predict(X_test[:, train_index])

                        # root mean squared errors
                        current_train_perf.append(
                            np.sqrt(mean_squared_error(y_app[train_index], y_pred_train)))
                        current_valid_perf.append(
                            np.sqrt(mean_squared_error(y_app[valid_index], y_pred_valid)))
                        current_test_perf.append(
                            np.sqrt(mean_squared_error(y_test, y_pred_test)))
                # For clcassification use SVM
                else:
                    KR = SVC(kernel='precomputed', **params_in)
                    # loop for each split on validation set level
                    # validation set level
                    for train_index, valid_index in inner_cv.split(X_app):
                        KR.fit(X_app[train_index, :]
                               [:, train_index], y_app[train_index])

                        # predict on the train, validation and test set
                        y_pred_train = KR.predict(
                            X_app[train_index, :][:, train_index])
                        y_pred_valid = KR.predict(
                            X_app[valid_index, :][:, train_index])
                        y_pred_test = KR.predict(X_test[:, train_index])

                        # root mean squared errors
                        current_train_perf.append(accuracy_score(
                            y_app[train_index], y_pred_train))
                        current_valid_perf.append(accuracy_score(
                            y_app[valid_index], y_pred_valid))
                        current_test_perf.append(
                            accuracy_score(y_test, y_pred_test))

                # average performance on inner splits
                train_pref[trial][index_out][index_in] = np.mean(
                    current_train_perf)
                val_pref[trial][index_out][index_in] = np.mean(
                    current_valid_perf)
                test_pref[trial][index_out][index_in] = np.mean(
                    current_test_perf)

                pbar.update(1)
    pbar.clear()

    print()
    print('4. Getting final performances...')
    # averages and confidences of performances on outer trials for each combination of parameters
    average_train_scores = np.mean(train_pref, axis=0)
    average_val_scores = np.mean(val_pref, axis=0)
    average_perf_scores = np.mean(test_pref, axis=0)
    # sample std is used here
    std_train_scores = np.std(train_pref, axis=0, ddof=1)
    std_val_scores = np.std(val_pref, axis=0, ddof=1)
    std_perf_scores = np.std(test_pref, axis=0, ddof=1)

    if model_type == 'regression':
        best_val_perf = np.amin(average_val_scores)
    else:
        best_val_perf = np.amax(average_val_scores)
    print()
    best_params_index = np.where(average_val_scores == best_val_perf)
    best_params_out = [param_list_precomputed[i] for i in best_params_index[0]]
    best_params_in = [param_list[i] for i in best_params_index[1]]
    # print('best_params_index: ', best_params_index)
    print('best_params_out: ', best_params_out)
    print('best_params_in: ', best_params_in)
    print()
    print('best_val_perf: ', best_val_perf)

    # below: only find one performance; muitiple pref might exist
    best_val_std = std_val_scores[best_params_index[0]
                                  [0]][best_params_index[1][0]]
    print('best_val_std: ', best_val_std)

    final_performance = average_perf_scores[best_params_index[0]
                                            [0]][best_params_index[1][0]]
    final_confidence = std_perf_scores[best_params_index[0]
                                       [0]][best_params_index[1][0]]
    print('final_performance: ', final_performance)
    print('final_confidence: ', final_confidence)
    train_performance = average_train_scores[best_params_index[0]
                                             [0]][best_params_index[1][0]]
    train_std = std_train_scores[best_params_index[0]
                                 [0]][best_params_index[1][0]]
    print('train_performance: ', train_performance)
    print('train_std: ', train_std)

    print()
    average_gram_matrix_time = np.mean(gram_matrix_time)
    std_gram_matrix_time = np.std(gram_matrix_time, ddof=1)
    best_gram_matrix_time = gram_matrix_time[best_params_index[0][0]]
    print('time to calculate gram matrix with different hyperpapams: {:.2f}±{:.2f}'
          .format(average_gram_matrix_time, std_gram_matrix_time))
    print('time to calculate best gram matrix: ', best_gram_matrix_time, 's')

    # print out as table.
    from collections import OrderedDict
    from tabulate import tabulate
    table_dict = {}
    if model_type == 'regression':
        for param_in in param_list:
            param_in['alpha'] = '{:.2e}'.format(param_in['alpha'])
    else:
        for param_in in param_list:
            param_in['C'] = '{:.2e}'.format(param_in['C'])
    table_dict['params'] = [{**param_out, **param_in}
                            for param_in in param_list for param_out in param_list_precomputed]
    table_dict['gram_matrix_time'] = ['{:.2f}'.format(gram_matrix_time[index_out])
                                      for param_in in param_list for index_out, _ in enumerate(param_list_precomputed)]
    table_dict['valid_perf'] = ['{:.2f}±{:.2f}'.format(average_val_scores[index_out][index_in], std_val_scores[index_out][index_in])
                                for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_precomputed)]
    table_dict['test_perf'] = ['{:.2f}±{:.2f}'.format(average_perf_scores[index_out][index_in], std_perf_scores[index_out][index_in])
                               for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_precomputed)]
    table_dict['train_perf'] = ['{:.2f}±{:.2f}'.format(average_train_scores[index_out][index_in], std_train_scores[index_out][index_in])
                                for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_precomputed)]
    keyorder = ['params', 'train_perf', 'valid_perf',
                'test_perf', 'gram_matrix_time']
    print()
    print(tabulate(OrderedDict(sorted(table_dict.items(),
                                      key=lambda i: keyorder.index(i[0]))), headers='keys'))
