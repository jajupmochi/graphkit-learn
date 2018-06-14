

def model_selection_for_precomputed_kernel(datafile, estimator,
                                           param_grid_precomputed, param_grid,
                                           model_type, NUM_TRIALS=30,
                                           datafile_y=None,
                                           extra_params=None,
                                           ds_name='ds-unknown'):
    """Perform model selection, fitting and testing for precomputed kernels using nested cv. Print out neccessary data during the process then finally the results.

    Parameters
    ----------
    datafile : string
        Path of dataset file.
    estimator : function
        kernel function used to estimate. This function needs to return a gram matrix.
    param_grid_precomputed : dictionary
        Dictionary with names (string) of parameters used to calculate gram matrices as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings. Params with length 1 will be omitted.
    param_grid : dictionary
        Dictionary with names (string) of parameters used as penelties as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings. Params with length 1 will be omitted.
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
    import os
    from os.path import basename, splitext
    from pygraph.utils.graphfiles import loadDataset
    from tqdm import tqdm
    tqdm.monitor_interval = 0

    results_dir = '../notebooks/results/' + estimator.__name__
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # open file to save all results for this dataset.
    with open(results_dir + '/' + ds_name + '.txt', 'w') as fresults:
        fresults.write('# This file contains results of ' + estimator.__name__ + ' on dataset ' + ds_name + ',\n# including gram matrices, serial numbers for gram matrix figures and performance.\n\n')

        # setup the model type
        model_type = model_type.lower()
        if model_type != 'regression' and model_type != 'classification':
            raise Exception(
                'The model type is incorrect! Please choose from regression or classification.')
        print()
        print('--- This is a %s problem ---' % model_type)
        fresults.write('This is a %s problem.\n\n' % model_type)

        # Load the dataset
        print()
        print('\nI. Loading dataset from file...')
        dataset, y = loadDataset(datafile, filename_y=datafile_y, extra_params=extra_params)

    #     import matplotlib.pyplot as plt      
    #     import networkx as nx
    #     nx.draw_networkx(dataset[30])
    #     plt.show()

        # Grid of parameters with a discrete number of values for each.
        param_list_precomputed = list(ParameterGrid(param_grid_precomputed))
        param_list = list(ParameterGrid(param_grid))
        # np.savetxt(results_name_pre + 'param_grid_precomputed.dt',
        #            [[key, value] for key, value in sorted(param_grid_precomputed)])
        # np.savetxt(results_name_pre + 'param_grid.dt',
        #            [[key, value] for key, value in sorted(param_grid)])

        gram_matrices = []  # a list to store gram matrices for all param_grid_precomputed
        gram_matrix_time = []  # a list to store time to calculate gram matrices
        param_list_pre_revised = [] # list to store param grids precomputed ignoring the useless ones

        # calculate all gram matrices
        print()
        print('2. Calculating gram matrices. This could take a while...')
        fresults.write('\nI. Gram matrices.\n\n')
        nb_gm_ignore = 0 # the number of gram matrices those should not be considered, as they may contain elements that are not numbers (NaN)
        for idx, params_out in enumerate(param_list_precomputed):
            rtn_data = estimator(dataset, **params_out)
            Kmatrix = rtn_data[0]
            current_run_time = rtn_data[1]
            if len(rtn_data) == 3:
                idx_trim = rtn_data[2] # the index of trimmed graph list
                y = [y[idx] for idx in idx_trim]
                
            Kmatrix_diag = Kmatrix.diagonal().copy()
            for i in range(len(Kmatrix)):
                for j in range(i, len(Kmatrix)):
    #                 if Kmatrix_diag[i] != 0 and Kmatrix_diag[j] != 0:
                    Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
                    Kmatrix[j][i] = Kmatrix[i][j]

            print()
            if params_out == {}:
                print('the gram matrix is: ')
                fresults.write('the gram matrix is:\n\n')
            else:
                print('the gram matrix with parameters', params_out, 'is: ')
                fresults.write('the gram matrix with parameters %s is:\n\n' % params_out)
            if np.isnan(Kmatrix).any(): # if the matrix contains elements that are not numbers
                nb_gm_ignore += 1
                print('ignored, as it contains elements that are not numbers.')
                fresults.write('ignored, as it contains elements that are not numbers.\n\n')
            else:
                print(Kmatrix)
                fresults.write(np.array2string(Kmatrix, separator=',', threshold=np.inf, floatmode='unique') + '\n\n')
                plt.matshow(Kmatrix)
                plt.colorbar()
                fig_file_name = results_dir + '/GM[ds]' + ds_name
                if params_out != {}:
                    fig_file_name += '[params]' + str(idx)
                plt.savefig(fig_file_name + '.eps', format='eps', dpi=300)
                plt.show()
                gram_matrices.append(Kmatrix)
                gram_matrix_time.append(current_run_time)
                param_list_pre_revised.append(params_out)
        print()
        print('{} gram matrices are calculated, {} of which are ignored.'.format(len(param_list_precomputed), nb_gm_ignore))
        fresults.write('{} gram matrices are calculated, {} of which are ignored.\n\n'.format(len(param_list_precomputed), nb_gm_ignore))
        fresults.write('serial numbers of gram matrix figure and their corresponding parameters settings:\n\n')
        fresults.write(''.join(['{}: {}\n'.format(idx, params_out)
                                            for idx, params_out in enumerate(param_list_precomputed)]))

        print()
        print('3. Fitting and predicting using nested cross validation. This could really take a while...')
        # Arrays to store scores
        train_pref = np.zeros(
            (NUM_TRIALS, len(param_list_pre_revised), len(param_list)))
        val_pref = np.zeros(
            (NUM_TRIALS, len(param_list_pre_revised), len(param_list)))
        test_pref = np.zeros(
            (NUM_TRIALS, len(param_list_pre_revised), len(param_list)))

        # Loop for each trial
        pbar = tqdm(total=NUM_TRIALS * len(param_list_pre_revised) * len(param_list),
                    desc='calculate performance', file=sys.stdout)
        for trial in range(NUM_TRIALS):  # Test set level
            # loop for each outer param tuple
            for index_out, params_out in enumerate(param_list_pre_revised):
                # split gram matrix and y to app and test sets.
                X_app, X_test, y_app, y_test = train_test_split(
                    gram_matrices[index_out], y, test_size=0.1)
                split_index_app = [y.index(y_i) for y_i in y_app if y_i in y]
                # split_index_test = [y.index(y_i) for y_i in y_test if y_i in y]
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
                    try:
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
                                y_pred_test = KR.predict(
                                    X_test[:, train_index])
    
                                # root mean squared errors
                                current_train_perf.append(accuracy_score(
                                    y_app[train_index], y_pred_train))
                                current_valid_perf.append(accuracy_score(
                                    y_app[valid_index], y_pred_valid))
                                current_test_perf.append(
                                    accuracy_score(y_test, y_pred_test))
                    except ValueError:
                        print(sys.exc_info()[0])
                        print(params_out, params_in)
    
                    # average performance on inner splits
                    train_pref[trial][index_out][index_in] = np.mean(
                        current_train_perf)
                    val_pref[trial][index_out][index_in] = np.mean(
                        current_valid_perf)
                    test_pref[trial][index_out][index_in] = np.mean(
                        current_test_perf)
    
                    pbar.update(1)
        pbar.clear()
        # np.save(results_name_pre + 'train_pref.dt', train_pref)
        # np.save(results_name_pre + 'val_pref.dt', val_pref)
        # np.save(results_name_pre + 'test_pref.dt', test_pref)

        print()
        print('4. Getting final performance...')
        fresults.write('\nII. Performance.\n\n')
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
        best_params_index = np.where(average_val_scores == best_val_perf)
        # find smallest val std with best val perf.
        best_val_stds = [std_val_scores[value][best_params_index[1][idx]] for idx, value in enumerate(best_params_index[0])]
        min_val_std = np.amin(best_val_stds)
        best_params_index = np.where(std_val_scores == min_val_std)
        best_params_out = [param_list_pre_revised[i] for i in best_params_index[0]]
        best_params_in = [param_list[i] for i in best_params_index[1]]
        print('best_params_out: ', best_params_out)
        print('best_params_in: ', best_params_in)
        print()
        print('best_val_perf: ', best_val_perf)
        print('best_val_std: ', min_val_std)
        fresults.write('best settings of hyper-params to build gram matrix: %s\n' % best_params_out)
        fresults.write('best settings of other hyper-params: %s\n\n' % best_params_in)
        fresults.write('best_val_perf: %s\n' % best_val_perf)
        fresults.write('best_val_std: %s\n' % min_val_std)

        final_performance = [average_perf_scores[value][best_params_index[1][idx]] for idx, value in enumerate(best_params_index[0])]
        final_confidence = [std_perf_scores[value][best_params_index[1][idx]] for idx, value in enumerate(best_params_index[0])]
        print('final_performance: ', final_performance)
        print('final_confidence: ', final_confidence)
        fresults.write('final_performance: %s\n' % final_performance)
        fresults.write('final_confidence: %s\n' % final_confidence)
        train_performance = [average_train_scores[value][best_params_index[1][idx]] for idx, value in enumerate(best_params_index[0])]
        train_std = [std_train_scores[value][best_params_index[1][idx]] for idx, value in enumerate(best_params_index[0])]
        print('train_performance: %s' % train_performance)
        print('train_std: ', train_std)
        fresults.write('train_performance: %s\n' % train_performance)
        fresults.write('train_std: %s\n\n' % train_std)

        print()
        average_gram_matrix_time = np.mean(gram_matrix_time)
        std_gram_matrix_time = np.std(gram_matrix_time, ddof=1)
        best_gram_matrix_time = [gram_matrix_time[i] for i in best_params_index[0]]
        ave_bgmt = np.mean(best_gram_matrix_time)
        std_bgmt = np.std(best_gram_matrix_time, ddof=1)
        print('time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s'
              .format(average_gram_matrix_time, std_gram_matrix_time))
        print('time to calculate best gram matrix: {:.2f}±{:.2f}s'.format(ave_bgmt, std_bgmt))
        fresults.write('time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s\n'
              .format(average_gram_matrix_time, std_gram_matrix_time))
        fresults.write('time to calculate best gram matrix: {:.2f}±{:.2f}s\n\n'.format(ave_bgmt, std_bgmt))

        # # save results to file
        # np.savetxt(results_name_pre + 'average_train_scores.dt',
        #            average_train_scores)
        # np.savetxt(results_name_pre + 'average_val_scores', average_val_scores)
        # np.savetxt(results_name_pre + 'average_perf_scores.dt',
        #            average_perf_scores)
        # np.savetxt(results_name_pre + 'std_train_scores.dt', std_train_scores)
        # np.savetxt(results_name_pre + 'std_val_scores.dt', std_val_scores)
        # np.savetxt(results_name_pre + 'std_perf_scores.dt', std_perf_scores)
    
        # np.save(results_name_pre + 'best_params_index', best_params_index)
        # np.save(results_name_pre + 'best_params_pre.dt', best_params_out)
        # np.save(results_name_pre + 'best_params_in.dt', best_params_in)
        # np.save(results_name_pre + 'best_val_perf.dt', best_val_perf)
        # np.save(results_name_pre + 'best_val_std.dt', best_val_std)
        # np.save(results_name_pre + 'final_performance.dt', final_performance)
        # np.save(results_name_pre + 'final_confidence.dt', final_confidence)
        # np.save(results_name_pre + 'train_performance.dt', train_performance)
        # np.save(results_name_pre + 'train_std.dt', train_std)
    
        # np.save(results_name_pre + 'gram_matrix_time.dt', gram_matrix_time)
        # np.save(results_name_pre + 'average_gram_matrix_time.dt',
        #         average_gram_matrix_time)
        # np.save(results_name_pre + 'std_gram_matrix_time.dt',
        #         std_gram_matrix_time)
        # np.save(results_name_pre + 'best_gram_matrix_time.dt',
        #         best_gram_matrix_time)
    
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
                                for param_in in param_list for param_out in param_list_pre_revised]
        table_dict['gram_matrix_time'] = ['{:.2f}'.format(gram_matrix_time[index_out])
                                          for param_in in param_list for index_out, _ in enumerate(param_list_pre_revised)]
        table_dict['valid_perf'] = ['{:.2f}±{:.2f}'.format(average_val_scores[index_out][index_in], std_val_scores[index_out][index_in])
                                    for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_pre_revised)]
        table_dict['test_perf'] = ['{:.2f}±{:.2f}'.format(average_perf_scores[index_out][index_in], std_perf_scores[index_out][index_in])
                                   for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_pre_revised)]
        table_dict['train_perf'] = ['{:.2f}±{:.2f}'.format(average_train_scores[index_out][index_in], std_train_scores[index_out][index_in])
                                    for index_in, _ in enumerate(param_list) for index_out, _ in enumerate(param_list_pre_revised)]
        keyorder = ['params', 'train_perf', 'valid_perf',
                    'test_perf', 'gram_matrix_time']
        print()
        tb_print = tabulate(OrderedDict(sorted(table_dict.items(),
                                          key=lambda i: keyorder.index(i[0]))), headers='keys')
        print(tb_print)
        fresults.write('table of performance v.s. hyper-params:\n\n%s\n\n' % tb_print)

        fresults.close()
