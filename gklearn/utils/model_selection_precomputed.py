import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, ParameterGrid

#from joblib import Parallel, delayed
from multiprocessing import Pool, Array
from functools import partial
import sys
import os
import time
import datetime
#from os.path import basename, splitext
from gklearn.utils.graphfiles import loadDataset
from tqdm import tqdm

#from memory_profiler import profile

#@profile
def model_selection_for_precomputed_kernel(datafile,
										   estimator,
										   param_grid_precomputed,
										   param_grid,
										   model_type,
										   NUM_TRIALS=30,
										   datafile_y=None,
										   extra_params=None,
										   ds_name='ds-unknown',
										   output_dir='outputs/',
										   n_jobs=1,
										   read_gm_from_file=False,
										   verbose=True):
	"""Perform model selection, fitting and testing for precomputed kernels 
	using nested CV. Print out neccessary data during the process then finally 
	the results.

	Parameters
	----------
	datafile : string
		Path of dataset file.
	estimator : function
		kernel function used to estimate. This function needs to return a gram matrix.
	param_grid_precomputed : dictionary
		Dictionary with names (string) of parameters used to calculate gram 
		matrices as keys and lists of parameter settings to try as values. This 
		enables searching over any sequence of parameter settings. Params with 
		length 1 will be omitted.
	param_grid : dictionary
		Dictionary with names (string) of parameters used as penelties as keys 
		and lists of parameter settings to try as values. This enables 
		searching over any sequence of parameter settings. Params with length 1
		will be omitted.
	model_type : string
		Type of the problem, can be 'regression' or 'classification'.
	NUM_TRIALS : integer
		Number of random trials of the outer CV loop. The default is 30.
	datafile_y : string
		Path of file storing y data. This parameter is optional depending on 
		the given dataset file.
	extra_params : dict
		Extra parameters for loading dataset. See function gklearn.utils.
		graphfiles.loadDataset for detail.
	ds_name : string
		Name of the dataset.
	n_jobs : int
		Number of jobs for parallelization.
	read_gm_from_file : boolean
		Whether gram matrices are loaded from a file.

	Examples
	--------
	>>> import numpy as np
	>>> from gklearn.utils.model_selection_precomputed import model_selection_for_precomputed_kernel
	>>> from gklearn.kernels.untilHPathKernel import untilhpathkernel
	>>>
	>>> datafile = '../datasets/MUTAG/MUTAG_A.txt'
	>>> estimator = untilhpathkernel
	>>> param_grid_precomputed = {’depth’:  np.linspace(1, 10, 10), ’k_func’:
			[’MinMax’, ’tanimoto’], ’compute_method’:  [’trie’]}
	>>> # ’C’ for classification problems and ’alpha’ for regression problems.
	>>> param_grid = [{’C’: np.logspace(-10, 10, num=41, base=10)}, {’alpha’:
			np.logspace(-10, 10, num=41, base=10)}]
	>>>
	>>> model_selection_for_precomputed_kernel(datafile, estimator, 
			param_grid_precomputed, param_grid[0], 'classification', ds_name=’MUTAG’)
	"""
	tqdm.monitor_interval = 0

	output_dir += estimator.__name__
	os.makedirs(output_dir, exist_ok=True)
	# a string to save all the results.
	str_fw = '###################### log time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '. ######################\n\n'
	str_fw += '# This file contains results of ' + estimator.__name__ + ' on dataset ' + ds_name + ',\n# including gram matrices, serial numbers for gram matrix figures and performance.\n\n'

	# setup the model type
	model_type = model_type.lower()
	if model_type != 'regression' and model_type != 'classification':
		raise Exception(
			'The model type is incorrect! Please choose from regression or classification.'
		)
	if verbose:
		print()
		print('--- This is a %s problem ---' % model_type)
	str_fw += 'This is a %s problem.\n' % model_type
	
	# calculate gram matrices rather than read them from file.
	if read_gm_from_file == False:
		# Load the dataset
		if verbose:
			print()
			print('\n1. Loading dataset from file...')
		if isinstance(datafile, str):
			dataset, y_all = loadDataset(
					datafile, filename_y=datafile_y, extra_params=extra_params)
		else: # load data directly from variable.
			dataset = datafile
			y_all = datafile_y				

		#	 import matplotlib.pyplot as plt
		#	 import networkx as nx
		#	 nx.draw_networkx(dataset[30])
		#	 plt.show()
	
		# Grid of parameters with a discrete number of values for each.
		param_list_precomputed = list(ParameterGrid(param_grid_precomputed))
		param_list = list(ParameterGrid(param_grid))
	
		gram_matrices = [
		]  # a list to store gram matrices for all param_grid_precomputed
		gram_matrix_time = [
		]  # a list to store time to calculate gram matrices
		param_list_pre_revised = [
		]  # list to store param grids precomputed ignoring the useless ones
	
		# calculate all gram matrices
		if verbose:
			print()
			print('2. Calculating gram matrices. This could take a while...')
		str_fw += '\nII. Gram matrices.\n\n'
		tts = time.time()  # start training time
		nb_gm_ignore = 0  # the number of gram matrices those should not be considered, as they may contain elements that are not numbers (NaN)
		for idx, params_out in enumerate(param_list_precomputed):
			y = y_all[:]
			params_out['n_jobs'] = n_jobs
			params_out['verbose'] = verbose
#			print(dataset)
#			import networkx as nx
#			nx.draw_networkx(dataset[1])
#			plt.show()
			rtn_data = estimator(dataset[:], **params_out)
			Kmatrix = rtn_data[0]
			current_run_time = rtn_data[1]
			# for some kernels, some graphs in datasets may not meet the 
			# kernels' requirements for graph structure. These graphs are trimmed. 
			if len(rtn_data) == 3:
				idx_trim = rtn_data[2]  # the index of trimmed graph list
				y = [y[idxt] for idxt in idx_trim] # trim y accordingly
#			Kmatrix = np.random.rand(2250, 2250)
#			current_run_time = 0.1
			
			# remove graphs whose kernels with themselves are zeros 
			# @todo: y not changed accordingly?
			Kmatrix_diag = Kmatrix.diagonal().copy()
			nb_g_ignore = 0
			for idxk, diag in enumerate(Kmatrix_diag):
				if diag == 0:
					Kmatrix = np.delete(Kmatrix, (idxk - nb_g_ignore), axis=0)
					Kmatrix = np.delete(Kmatrix, (idxk - nb_g_ignore), axis=1)
					nb_g_ignore += 1
			# normalization
			# @todo: works only for undirected graph?
			Kmatrix_diag = Kmatrix.diagonal().copy()
			for i in range(len(Kmatrix)):
				for j in range(i, len(Kmatrix)):
					Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
					Kmatrix[j][i] = Kmatrix[i][j]
			if verbose:
				print()
			if params_out == {}:
				if verbose:
					print('the gram matrix is: ')
				str_fw += 'the gram matrix is:\n\n'
			else:
				if verbose:
					print('the gram matrix with parameters', params_out, 'is: \n\n')
				str_fw += 'the gram matrix with parameters %s is:\n\n' % params_out
			if len(Kmatrix) < 2:
				nb_gm_ignore += 1
				if verbose:
					print('ignored, as at most only one of all its diagonal value is non-zero.')
				str_fw += 'ignored, as at most only one of all its diagonal value is non-zero.\n\n'
			else:				
				if np.isnan(Kmatrix).any(
				):  # if the matrix contains elements that are not numbers
					nb_gm_ignore += 1
					if verbose:
						print('ignored, as it contains elements that are not numbers.')
					str_fw += 'ignored, as it contains elements that are not numbers.\n\n'
				else:
#					print(Kmatrix)
					str_fw += np.array2string(
							Kmatrix,
							separator=',') + '\n\n'
#							separator=',',
#							threshold=np.inf,
#							floatmode='unique') + '\n\n'

					fig_file_name = output_dir + '/GM[ds]' + ds_name
					if params_out != {}:
						fig_file_name += '[params]' + str(idx)
					plt.imshow(Kmatrix)
					plt.colorbar()
					plt.savefig(fig_file_name + '.eps', format='eps', dpi=300)
#					plt.show()
					plt.clf()
					gram_matrices.append(Kmatrix)
					gram_matrix_time.append(current_run_time)
					param_list_pre_revised.append(params_out)
					if nb_g_ignore > 0:
						if verbose:
							print(', where %d graphs are ignored as their graph kernels with themselves are zeros.' % nb_g_ignore)
						str_fw += ', where %d graphs are ignored as their graph kernels with themselves are zeros.' % nb_g_ignore
		if verbose:
			print()
			print(
			'{} gram matrices are calculated, {} of which are ignored.'.format(
				len(param_list_precomputed), nb_gm_ignore))
		str_fw += '{} gram matrices are calculated, {} of which are ignored.\n\n'.format(len(param_list_precomputed), nb_gm_ignore)
		str_fw += 'serial numbers of gram matrix figures and their corresponding parameters settings:\n\n'
		str_fw += ''.join([
			'{}: {}\n'.format(idx, params_out)
			for idx, params_out in enumerate(param_list_precomputed)
		])

		if verbose:
			print()
		if len(gram_matrices) == 0:
			if verbose:
				print('all gram matrices are ignored, no results obtained.')
			str_fw += '\nall gram matrices are ignored, no results obtained.\n\n'
		else:
			# save gram matrices to file.
#			np.savez(output_dir + '/' + ds_name + '.gm', 
#					 gms=gram_matrices, params=param_list_pre_revised, y=y, 
#					 gmtime=gram_matrix_time)
			if verbose:
				print(
				'3. Fitting and predicting using nested cross validation. This could really take a while...'
				)
			
			# ---- use pool.imap_unordered to parallel and track progress. ----
#			train_pref = []
#			val_pref = []
#			test_pref = []
#			def func_assign(result, var_to_assign):
#				for idx, itm in enumerate(var_to_assign):
#					itm.append(result[idx])				
#			trial_do_partial = partial(trial_do, param_list_pre_revised, param_list, y, model_type)
#					  
#			parallel_me(trial_do_partial, range(NUM_TRIALS), func_assign, 
#						[train_pref, val_pref, test_pref], glbv=gram_matrices,
#						method='imap_unordered', n_jobs=n_jobs, chunksize=1,
#						itr_desc='cross validation')
			
			def init_worker(gms_toshare):
				global G_gms
				G_gms = gms_toshare
			
#			gram_matrices = np.array(gram_matrices)
#			gms_shape = gram_matrices.shape
#			gms_array = Array('d', np.reshape(gram_matrices.copy(), -1, order='C'))
#			pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(gms_array, gms_shape))
			pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(gram_matrices,))
			trial_do_partial = partial(parallel_trial_do, param_list_pre_revised, param_list, y, model_type)
			train_pref = []
			val_pref = []
			test_pref = []
#			if NUM_TRIALS < 1000 * n_jobs:
#				chunksize = int(NUM_TRIALS / n_jobs) + 1
#			else:
#				chunksize = 1000
			chunksize = 1
			if verbose:
				iterator = tqdm(pool.imap_unordered(trial_do_partial, 
						range(NUM_TRIALS), chunksize), desc='cross validation', file=sys.stdout)
			else:
				iterator = pool.imap_unordered(trial_do_partial, range(NUM_TRIALS), chunksize)
			for o1, o2, o3 in iterator:
				train_pref.append(o1)
				val_pref.append(o2)
				test_pref.append(o3)
			pool.close()
			pool.join()
	
#			# ---- use pool.map to parallel. ----
#			pool =  Pool(n_jobs)
#			trial_do_partial = partial(trial_do, param_list_pre_revised, param_list, gram_matrices, y[0:250], model_type)
#			result_perf = pool.map(trial_do_partial, range(NUM_TRIALS))
#			train_pref = [item[0] for item in result_perf]
#			val_pref = [item[1] for item in result_perf]
#			test_pref = [item[2] for item in result_perf]
	
#			# ---- direct running, normally use a single CPU core. ----
#			train_pref = []
#			val_pref = []
#			test_pref = []
#			for i in tqdm(range(NUM_TRIALS), desc='cross validation', file=sys.stdout):
#				o1, o2, o3 = trial_do(param_list_pre_revised, param_list, gram_matrices, y, model_type, i)
#				train_pref.append(o1)
#				val_pref.append(o2)
#				test_pref.append(o3)
#			print()
	
			if verbose:
				print()
				print('4. Getting final performance...')
			str_fw += '\nIII. Performance.\n\n'
			# averages and confidences of performances on outer trials for each combination of parameters
			average_train_scores = np.mean(train_pref, axis=0)
#			print('val_pref: ', val_pref[0][0])
			average_val_scores = np.mean(val_pref, axis=0)
#			print('test_pref: ', test_pref[0][0])
			average_perf_scores = np.mean(test_pref, axis=0)
			# sample std is used here
			std_train_scores = np.std(train_pref, axis=0, ddof=1)
			std_val_scores = np.std(val_pref, axis=0, ddof=1)
			std_perf_scores = np.std(test_pref, axis=0, ddof=1)
	
			if model_type == 'regression':
				best_val_perf = np.amin(average_val_scores)
			else:
				best_val_perf = np.amax(average_val_scores)
#			print('average_val_scores: ', average_val_scores)
#			print('best_val_perf: ', best_val_perf)
#			print()
			best_params_index = np.where(average_val_scores == best_val_perf)
			# find smallest val std with best val perf.
			best_val_stds = [
				std_val_scores[value][best_params_index[1][idx]]
				for idx, value in enumerate(best_params_index[0])
			]
			min_val_std = np.amin(best_val_stds)
			best_params_index = np.where(std_val_scores == min_val_std)
			best_params_out = [
				param_list_pre_revised[i] for i in best_params_index[0]
			]
			best_params_in = [param_list[i] for i in best_params_index[1]]
			if verbose:
				print('best_params_out: ', best_params_out)
				print('best_params_in: ', best_params_in)
				print()
				print('best_val_perf: ', best_val_perf)
				print('best_val_std: ', min_val_std)
			str_fw += 'best settings of hyper-params to build gram matrix: %s\n' % best_params_out
			str_fw += 'best settings of other hyper-params: %s\n\n' % best_params_in
			str_fw += 'best_val_perf: %s\n' % best_val_perf
			str_fw += 'best_val_std: %s\n' % min_val_std
	
#			print(best_params_index)
#			print(best_params_index[0])
#			print(average_perf_scores)
			final_performance = [
				average_perf_scores[value][best_params_index[1][idx]]
				for idx, value in enumerate(best_params_index[0])
			]
			final_confidence = [
				std_perf_scores[value][best_params_index[1][idx]]
				for idx, value in enumerate(best_params_index[0])
			]
			if verbose:
				print('final_performance: ', final_performance)
				print('final_confidence: ', final_confidence)
			str_fw += 'final_performance: %s\n' % final_performance
			str_fw += 'final_confidence: %s\n' % final_confidence
			train_performance = [
				average_train_scores[value][best_params_index[1][idx]]
				for idx, value in enumerate(best_params_index[0])
			]
			train_std = [
				std_train_scores[value][best_params_index[1][idx]]
				for idx, value in enumerate(best_params_index[0])
			]
			if verbose:
				print('train_performance: %s' % train_performance)
				print('train_std: ', train_std)
			str_fw += 'train_performance: %s\n' % train_performance
			str_fw += 'train_std: %s\n\n' % train_std

			if verbose:
				print()
			tt_total = time.time() - tts  # training time for all hyper-parameters
			average_gram_matrix_time = np.mean(gram_matrix_time)
			std_gram_matrix_time = np.std(gram_matrix_time, ddof=1) if len(gram_matrix_time) > 1 else 0
			best_gram_matrix_time = [
				gram_matrix_time[i] for i in best_params_index[0]
			]
			ave_bgmt = np.mean(best_gram_matrix_time)
			std_bgmt = np.std(best_gram_matrix_time, ddof=1) if len(best_gram_matrix_time) > 1 else 0
			if verbose:
				print('time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s'
					  .format(average_gram_matrix_time, std_gram_matrix_time))
				print('time to calculate best gram matrix: {:.2f}±{:.2f}s'.format(
						ave_bgmt, std_bgmt))
				print('total training time with all hyper-param choices: {:.2f}s'.format(
						tt_total))
			str_fw += 'time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s\n'.format(average_gram_matrix_time, std_gram_matrix_time)
			str_fw += 'time to calculate best gram matrix: {:.2f}±{:.2f}s\n'.format(ave_bgmt, std_bgmt)
			str_fw += 'total training time with all hyper-param choices: {:.2f}s\n\n'.format(tt_total)
	
			# # save results to file
			# np.savetxt(results_name_pre + 'average_train_scores.dt',
			#			average_train_scores)
			# np.savetxt(results_name_pre + 'average_val_scores', average_val_scores)
			# np.savetxt(results_name_pre + 'average_perf_scores.dt',
			#			average_perf_scores)
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
			#		 average_gram_matrix_time)
			# np.save(results_name_pre + 'std_gram_matrix_time.dt',
			#		 std_gram_matrix_time)
			# np.save(results_name_pre + 'best_gram_matrix_time.dt',
			#		 best_gram_matrix_time)
	
	# read gram matrices from file.
	else:	
		# Grid of parameters with a discrete number of values for each.
#		param_list_precomputed = list(ParameterGrid(param_grid_precomputed))
		param_list = list(ParameterGrid(param_grid))
	
		# read gram matrices from file.
		if verbose:
			print()
			print('2. Reading gram matrices from file...')
		str_fw += '\nII. Gram matrices.\n\nGram matrices are read from file, see last log for detail.\n'
		gmfile = np.load(output_dir + '/' + ds_name + '.gm.npz')
		gram_matrices = gmfile['gms'] # a list to store gram matrices for all param_grid_precomputed
		gram_matrix_time = gmfile['gmtime'] # time used to compute the gram matrices
		param_list_pre_revised = gmfile['params'] # list to store param grids precomputed ignoring the useless ones
		y = gmfile['y'].tolist()
		
		tts = time.time()  # start training time
#		nb_gm_ignore = 0  # the number of gram matrices those should not be considered, as they may contain elements that are not numbers (NaN)			
		if verbose:
			print(
					'3. Fitting and predicting using nested cross validation. This could really take a while...'
					)
 
		# ---- use pool.imap_unordered to parallel and track progress. ----
		def init_worker(gms_toshare):
			global G_gms
			G_gms = gms_toshare

		pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(gram_matrices,))
		trial_do_partial = partial(parallel_trial_do, param_list_pre_revised, param_list, y, model_type)
		train_pref = []
		val_pref = []
		test_pref = []
		chunksize = 1
		if verbose:
			iterator = tqdm(pool.imap_unordered(trial_do_partial, 
					range(NUM_TRIALS), chunksize), desc='cross validation', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(trial_do_partial, range(NUM_TRIALS), chunksize)
		for o1, o2, o3 in iterator:
			train_pref.append(o1)
			val_pref.append(o2)
			test_pref.append(o3)
		pool.close()
		pool.join()
		
		# # ---- use pool.map to parallel. ----
		# result_perf = pool.map(trial_do_partial, range(NUM_TRIALS))
		# train_pref = [item[0] for item in result_perf]
		# val_pref = [item[1] for item in result_perf]
		# test_pref = [item[2] for item in result_perf]

		# # ---- use joblib.Parallel to parallel and track progress. ----
		# trial_do_partial = partial(trial_do, param_list_pre_revised, param_list, gram_matrices, y, model_type)
		# result_perf = Parallel(n_jobs=n_jobs, verbose=10)(delayed(trial_do_partial)(trial) for trial in range(NUM_TRIALS))
		# train_pref = [item[0] for item in result_perf]
		# val_pref = [item[1] for item in result_perf]
		# test_pref = [item[2] for item in result_perf]

#		# ---- direct running, normally use a single CPU core. ----
#		train_pref = []
#		val_pref = []
#		test_pref = []
#		for i in tqdm(range(NUM_TRIALS), desc='cross validation', file=sys.stdout):
#			o1, o2, o3 = trial_do(param_list_pre_revised, param_list, gram_matrices, y, model_type, i)
#			train_pref.append(o1)
#			val_pref.append(o2)
#			test_pref.append(o3)

		if verbose:
			print()
			print('4. Getting final performance...')
		str_fw += '\nIII. Performance.\n\n'
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
		best_val_stds = [
			std_val_scores[value][best_params_index[1][idx]]
			for idx, value in enumerate(best_params_index[0])
		]
		min_val_std = np.amin(best_val_stds)
		best_params_index = np.where(std_val_scores == min_val_std)
		best_params_out = [
			param_list_pre_revised[i] for i in best_params_index[0]
		]
		best_params_in = [param_list[i] for i in best_params_index[1]]
		if verbose:
			print('best_params_out: ', best_params_out)
			print('best_params_in: ', best_params_in)
			print()
			print('best_val_perf: ', best_val_perf)
			print('best_val_std: ', min_val_std)
		str_fw += 'best settings of hyper-params to build gram matrix: %s\n' % best_params_out
		str_fw += 'best settings of other hyper-params: %s\n\n' % best_params_in
		str_fw += 'best_val_perf: %s\n' % best_val_perf
		str_fw += 'best_val_std: %s\n' % min_val_std

		final_performance = [
			average_perf_scores[value][best_params_index[1][idx]]
			for idx, value in enumerate(best_params_index[0])
		]
		final_confidence = [
			std_perf_scores[value][best_params_index[1][idx]]
			for idx, value in enumerate(best_params_index[0])
		]
		if verbose:
			print('final_performance: ', final_performance)
			print('final_confidence: ', final_confidence)
		str_fw += 'final_performance: %s\n' % final_performance
		str_fw += 'final_confidence: %s\n' % final_confidence
		train_performance = [
			average_train_scores[value][best_params_index[1][idx]]
			for idx, value in enumerate(best_params_index[0])
		]
		train_std = [
			std_train_scores[value][best_params_index[1][idx]]
			for idx, value in enumerate(best_params_index[0])
		]
		if verbose:
			print('train_performance: %s' % train_performance)
			print('train_std: ', train_std)
		str_fw += 'train_performance: %s\n' % train_performance
		str_fw += 'train_std: %s\n\n' % train_std

		if verbose:
			print()
		average_gram_matrix_time = np.mean(gram_matrix_time)
		std_gram_matrix_time = np.std(gram_matrix_time, ddof=1) if len(gram_matrix_time) > 1 else 0
		best_gram_matrix_time = [
			gram_matrix_time[i] for i in best_params_index[0]
		]
		ave_bgmt = np.mean(best_gram_matrix_time)
		std_bgmt = np.std(best_gram_matrix_time, ddof=1) if len(best_gram_matrix_time) > 1 else 0
		if verbose:		
			print(
					'time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s'
					.format(average_gram_matrix_time, std_gram_matrix_time))
			print('time to calculate best gram matrix: {:.2f}±{:.2f}s'.format(
					ave_bgmt, std_bgmt))
		tt_poster = time.time() - tts  # training time with hyper-param choices who did not participate in calculation of gram matrices
		if verbose:
			print(
					'training time with hyper-param choices who did not participate in calculation of gram matrices: {:.2f}s'.format(
							tt_poster))
			print('total training time with all hyper-param choices: {:.2f}s'.format(
					tt_poster + np.sum(gram_matrix_time)))
#		str_fw += 'time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s\n'.format(average_gram_matrix_time, std_gram_matrix_time)
#		str_fw += 'time to calculate best gram matrix: {:.2f}±{:.2f}s\n'.format(ave_bgmt, std_bgmt)
		str_fw += 'training time with hyper-param choices who did not participate in calculation of gram matrices: {:.2f}s\n\n'.format(tt_poster)

		# open file to save all results for this dataset.
		os.makedirs(output_dir, exist_ok=True)
			
	# print out results as table.
	str_fw += printResultsInTable(param_list, param_list_pre_revised, average_val_scores,
			  std_val_scores, average_perf_scores, std_perf_scores,
			  average_train_scores, std_train_scores, gram_matrix_time,
			  model_type, verbose)
			
	# open file to save all results for this dataset.
	if not os.path.exists(output_dir + '/' + ds_name + '.output.txt'):
		with open(output_dir + '/' + ds_name + '.output.txt', 'w') as f:
			f.write(str_fw)
	else:
		with open(output_dir + '/' + ds_name + '.output.txt', 'r+') as f:
			content = f.read()
			f.seek(0, 0)
			f.write(str_fw + '\n\n\n' + content)
			
	return final_performance, final_confidence


def trial_do(param_list_pre_revised, param_list, gram_matrices, y, model_type, trial): # Test set level

#	# get gram matrices from global variables.
#	gram_matrices = np.reshape(G_gms.copy(), G_gms_shape, order='C')
	
	# Arrays to store scores
	train_pref = np.zeros((len(param_list_pre_revised), len(param_list)))
	val_pref = np.zeros((len(param_list_pre_revised), len(param_list)))
	test_pref = np.zeros((len(param_list_pre_revised), len(param_list)))

	# randomness added to seeds of split function below. "high" is "size" times
	# 10 so that at least 10 different random output will be yielded. Remove
	# these lines if identical outputs is required.
	rdm_out = np.random.RandomState(seed=None)
	rdm_seed_out_l = rdm_out.uniform(high=len(param_list_pre_revised) * 10, 
								   size=len(param_list_pre_revised))
#	print(trial, rdm_seed_out_l)
#	print()
	# loop for each outer param tuple
	for index_out, params_out in enumerate(param_list_pre_revised):
		# get gram matrices from global variables.
#		gm_now = G_gms[index_out * G_gms_shape[1] * G_gms_shape[2]:(index_out + 1) * G_gms_shape[1] * G_gms_shape[2]]
#		gm_now = np.reshape(gm_now.copy(), (G_gms_shape[1], G_gms_shape[2]), order='C')
		gm_now = gram_matrices[index_out].copy()
	
		# split gram matrix and y to app and test sets.
		indices = range(len(y))
		# The argument "random_state" in function "train_test_split" can not be
		# set to None, because it will use RandomState instance used by 
		# np.random, which is possible for multiple subprocesses to inherit the
		# same seed if they forked at the same time, leading to identical 
		# random variates for different subprocesses. Instead, we use "trial" 
		# and "index_out" parameters to generate different seeds for different 
		# trials/subprocesses and outer loops. "rdm_seed_out_l" is used to add 
		# randomness into seeds, so that it yields a different output every 
		# time the program is run. To yield identical outputs every time,
		# remove the second line below. Same method is used to the "KFold"
		# function in the inner loop.
		rdm_seed_out = (trial + 1) * (index_out + 1)
		rdm_seed_out = (rdm_seed_out + int(rdm_seed_out_l[index_out])) % (2 ** 32 - 1)
#		print(trial, rdm_seed_out)
		X_app, X_test, y_app, y_test, idx_app, idx_test = train_test_split(
			gm_now, y, indices, test_size=0.1, 
			random_state=rdm_seed_out, shuffle=True)
#		print(trial, idx_app, idx_test)
#		print()
		X_app = X_app[:, idx_app]
		X_test = X_test[:, idx_app]
		y_app = np.array(y_app)
		y_test = np.array(y_test)

		rdm_seed_in_l = rdm_out.uniform(high=len(param_list) * 10, 
								   size=len(param_list))
		# loop for each inner param tuple
		for index_in, params_in in enumerate(param_list):
#			if trial == 0:
#				print(index_out, index_in)
#				print('params_in: ', params_in)
#			st = time.time()
			rdm_seed_in = (trial + 1) * (index_out + 1) * (index_in + 1)
#			print("rdm_seed_in1: ", trial, index_in, rdm_seed_in)
			rdm_seed_in = (rdm_seed_in + int(rdm_seed_in_l[index_in])) % (2 ** 32 - 1)
#			print("rdm_seed_in2: ", trial, index_in, rdm_seed_in)
			inner_cv = KFold(n_splits=10, shuffle=True, random_state=rdm_seed_in)
			current_train_perf = []
			current_valid_perf = []
			current_test_perf = [] 

			# For regression use the Kernel Ridge method
#			try:
			if model_type == 'regression':
				kr = KernelRidge(kernel='precomputed', **params_in)
				# loop for each split on validation set level
				# validation set level
				for train_index, valid_index in inner_cv.split(X_app):
#					print("train_index, valid_index: ", trial, index_in, train_index, valid_index)
#					if trial == 0:
#						print('train_index: ', train_index)
#						print('valid_index: ', valid_index)
#						print('idx_test: ', idx_test)
#						print('y_app[train_index]: ', y_app[train_index])
#						print('X_app[train_index, :][:, train_index]: ', X_app[train_index, :][:, train_index])
#						print('X_app[valid_index, :][:, train_index]: ', X_app[valid_index, :][:, train_index])
					kr.fit(X_app[train_index, :][:, train_index],
						   y_app[train_index])

					# predict on the train, validation and test set
					y_pred_train = kr.predict(
						X_app[train_index, :][:, train_index])
					y_pred_valid = kr.predict(
						X_app[valid_index, :][:, train_index])
#					if trial == 0:	 
#						print('y_pred_valid: ', y_pred_valid)
#						print()
					y_pred_test = kr.predict(
						X_test[:, train_index])

					# root mean squared errors
					current_train_perf.append(
						np.sqrt(
							mean_squared_error(
								y_app[train_index], y_pred_train)))
					current_valid_perf.append(
						np.sqrt(
							mean_squared_error(
								y_app[valid_index], y_pred_valid)))
#					if trial == 0:
#						print(mean_squared_error(
#								y_app[valid_index], y_pred_valid))
					current_test_perf.append(
						np.sqrt(
							mean_squared_error(
								y_test, y_pred_test)))
			# For clcassification use SVM
			else:
				svc = SVC(kernel='precomputed', cache_size=200, 
						  verbose=False, **params_in)
				# loop for each split on validation set level
				# validation set level
				for train_index, valid_index in inner_cv.split(X_app):
#						np.savez("bug.npy",X_app[train_index, :][:, train_index],y_app[train_index])
#					if trial == 0:
#						print('train_index: ', train_index)
#						print('valid_index: ', valid_index)
#						print('idx_test: ', idx_test)
#						print('y_app[train_index]: ', y_app[train_index])
#						print('X_app[train_index, :][:, train_index]: ', X_app[train_index, :][:, train_index])
#						print('X_app[valid_index, :][:, train_index]: ', X_app[valid_index, :][:, train_index])
					svc.fit(X_app[train_index, :][:, train_index],
						   y_app[train_index])
					
					# predict on the train, validation and test set
					y_pred_train = svc.predict(
						X_app[train_index, :][:, train_index])
					y_pred_valid = svc.predict(
						X_app[valid_index, :][:, train_index])
					y_pred_test = svc.predict(
						X_test[:, train_index])

					# root mean squared errors
					current_train_perf.append(
						accuracy_score(y_app[train_index],
									   y_pred_train))
					current_valid_perf.append(
						accuracy_score(y_app[valid_index],
									   y_pred_valid))
					current_test_perf.append(
						accuracy_score(y_test, y_pred_test))
#			except ValueError:
#				print(sys.exc_info()[0])
#				print(params_out, params_in)

			# average performance on inner splits
			train_pref[index_out][index_in] = np.mean(
				current_train_perf)
			val_pref[index_out][index_in] = np.mean(
				current_valid_perf)
			test_pref[index_out][index_in] = np.mean(
				current_test_perf)
#			print(time.time() - st)
#	if trial == 0:
#		print('val_pref: ', val_pref)
#		print('test_pref: ', test_pref)

	return train_pref, val_pref, test_pref

def parallel_trial_do(param_list_pre_revised, param_list, y, model_type, trial):
	train_pref, val_pref, test_pref = trial_do(param_list_pre_revised, 
											   param_list, G_gms, y, 
											   model_type, trial)
	return train_pref, val_pref, test_pref


def compute_gram_matrices(dataset, y, estimator, param_list_precomputed, 
						  output_dir, ds_name,
						  n_jobs=1, str_fw='', verbose=True):
	gram_matrices = [
		]  # a list to store gram matrices for all param_grid_precomputed
	gram_matrix_time = [
		]  # a list to store time to calculate gram matrices
	param_list_pre_revised = [
		]  # list to store param grids precomputed ignoring the useless ones
	
	nb_gm_ignore = 0  # the number of gram matrices those should not be considered, as they may contain elements that are not numbers (NaN)
	for idx, params_out in enumerate(param_list_precomputed):
		params_out['n_jobs'] = n_jobs
#			print(dataset)
#			import networkx as nx
#			nx.draw_networkx(dataset[1])
#			plt.show()
		rtn_data = estimator(dataset[:], **params_out)
		Kmatrix = rtn_data[0]
		current_run_time = rtn_data[1]
		# for some kernels, some graphs in datasets may not meet the 
		# kernels' requirements for graph structure. These graphs are trimmed. 
		if len(rtn_data) == 3:
			idx_trim = rtn_data[2]  # the index of trimmed graph list
			y = [y[idxt] for idxt in idx_trim] # trim y accordingly

		Kmatrix_diag = Kmatrix.diagonal().copy()
		# remove graphs whose kernels with themselves are zeros
		nb_g_ignore = 0
		for idxk, diag in enumerate(Kmatrix_diag):
			if diag == 0:
				Kmatrix = np.delete(Kmatrix, (idxk - nb_g_ignore), axis=0)
				Kmatrix = np.delete(Kmatrix, (idxk - nb_g_ignore), axis=1)
				nb_g_ignore += 1
		# normalization
		for i in range(len(Kmatrix)):
			for j in range(i, len(Kmatrix)):
				Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
				Kmatrix[j][i] = Kmatrix[i][j]

		if verbose:
			print()
		if params_out == {}:
			if verbose:
				print('the gram matrix is: ')
			str_fw += 'the gram matrix is:\n\n'
		else:
			if verbose:
				print('the gram matrix with parameters', params_out, 'is: ')
			str_fw += 'the gram matrix with parameters %s is:\n\n' % params_out
		if len(Kmatrix) < 2:
			nb_gm_ignore += 1
			if verbose:
				print('ignored, as at most only one of all its diagonal value is non-zero.')
			str_fw += 'ignored, as at most only one of all its diagonal value is non-zero.\n\n'
		else:				
			if np.isnan(Kmatrix).any(
			):  # if the matrix contains elements that are not numbers
				nb_gm_ignore += 1
				if verbose:
					print('ignored, as it contains elements that are not numbers.')
				str_fw += 'ignored, as it contains elements that are not numbers.\n\n'
			else:
#					print(Kmatrix)
				str_fw += np.array2string(
						Kmatrix,
						separator=',') + '\n\n'
#							separator=',',
#							threshold=np.inf,
#							floatmode='unique') + '\n\n'

				fig_file_name = output_dir + '/GM[ds]' + ds_name
				if params_out != {}:
					fig_file_name += '[params]' + str(idx)
				plt.imshow(Kmatrix)
				plt.colorbar()
				plt.savefig(fig_file_name + '.eps', format='eps', dpi=300)
#					plt.show()
				plt.clf()
				gram_matrices.append(Kmatrix)
				gram_matrix_time.append(current_run_time)
				param_list_pre_revised.append(params_out)
				if nb_g_ignore > 0:
					if verbose:
						print(', where %d graphs are ignored as their graph kernels with themselves are zeros.' % nb_g_ignore)
					str_fw += ', where %d graphs are ignored as their graph kernels with themselves are zeros.' % nb_g_ignore
	if verbose:
		print()
		print(
			'{} gram matrices are calculated, {} of which are ignored.'.format(
				len(param_list_precomputed), nb_gm_ignore))
	str_fw += '{} gram matrices are calculated, {} of which are ignored.\n\n'.format(len(param_list_precomputed), nb_gm_ignore)
	str_fw += 'serial numbers of gram matrix figures and their corresponding parameters settings:\n\n'
	str_fw += ''.join([
		'{}: {}\n'.format(idx, params_out)
		for idx, params_out in enumerate(param_list_precomputed)
	])
			
	return gram_matrices, gram_matrix_time, param_list_pre_revised, y, str_fw


def read_gram_matrices_from_file(output_dir, ds_name):
	gmfile = np.load(output_dir + '/' + ds_name + '.gm.npz')
	gram_matrices = gmfile['gms'] # a list to store gram matrices for all param_grid_precomputed
	param_list_pre_revised = gmfile['params'] # list to store param grids precomputed ignoring the useless ones
	y = gmfile['y'].tolist()
	return gram_matrices, param_list_pre_revised, y


def printResultsInTable(param_list, param_list_pre_revised, average_val_scores,
						std_val_scores, average_perf_scores, std_perf_scores,
						average_train_scores, std_train_scores, gram_matrix_time,
						model_type, verbose):
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
	table_dict['gram_matrix_time'] = [
		'{:.2f}'.format(gram_matrix_time[index_out])
		for param_in in param_list
		for index_out, _ in enumerate(param_list_pre_revised)
	]
	table_dict['valid_perf'] = [
		'{:.2f}±{:.2f}'.format(average_val_scores[index_out][index_in],
							   std_val_scores[index_out][index_in])
		for index_in, _ in enumerate(param_list)
		for index_out, _ in enumerate(param_list_pre_revised)
	]
	table_dict['test_perf'] = [
		'{:.2f}±{:.2f}'.format(average_perf_scores[index_out][index_in],
							   std_perf_scores[index_out][index_in])
		for index_in, _ in enumerate(param_list)
		for index_out, _ in enumerate(param_list_pre_revised)
	]
	table_dict['train_perf'] = [
		'{:.2f}±{:.2f}'.format(average_train_scores[index_out][index_in],
							   std_train_scores[index_out][index_in])
		for index_in, _ in enumerate(param_list)
		for index_out, _ in enumerate(param_list_pre_revised)
	]
	
	keyorder = [
		'params', 'train_perf', 'valid_perf', 'test_perf',
		'gram_matrix_time'
	]
	if verbose:
		print()
	tb_print = tabulate(OrderedDict(sorted(table_dict.items(), 
						key=lambda i: keyorder.index(i[0]))), headers='keys')
#			print(tb_print)
	return 'table of performance v.s. hyper-params:\n\n%s\n\n' % tb_print