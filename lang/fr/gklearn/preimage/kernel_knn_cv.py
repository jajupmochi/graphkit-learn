#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:52:15 2020

@author: ljia
"""
import numpy as np
import csv
import os
import os.path
from gklearn.utils import Dataset
from sklearn.model_selection import ShuffleSplit
from gklearn.preimage import MedianPreimageGenerator
from gklearn.utils import normalize_gram_matrix, compute_distance_matrix
from gklearn.preimage.utils import get_same_item_indices
from gklearn.utils.knn import knn_classification
from gklearn.preimage.utils import compute_k_dis
	

def kernel_knn_cv(ds_name, train_examples, knn_options, mpg_options, kernel_options, ged_options, mge_options, save_results=True, load_gm='auto', dir_save='', irrelevant_labels=None, edge_required=False, cut_range=None):
	
	# 1. get dataset.
	print('1. getting dataset...')
	dataset_all = Dataset()
	dataset_all.load_predefined_dataset(ds_name)
	dataset_all.trim_dataset(edge_required=edge_required)
	if irrelevant_labels is not None:
		dataset_all.remove_labels(**irrelevant_labels)
	if cut_range is not None:
		dataset_all.cut_graphs(cut_range)

	if save_results:
		# create result files.
		print('creating output files...')
		fn_output_detail, fn_output_summary = _init_output_file_knn(ds_name, kernel_options['name'], mpg_options['fit_method'], dir_save)
	else:
		fn_output_detail, fn_output_summary = None, None
		
	# 2. compute/load Gram matrix a priori.
	print('2. computing/loading Gram matrix...')
	gram_matrix_unnorm, time_precompute_gm = _get_gram_matrix(load_gm, dir_save, ds_name, kernel_options, dataset_all)
	
	# 3. perform k-nn CV.
	print('3. performing k-nn CV...')
	if train_examples == 'k-graphs' or train_examples == 'expert' or train_examples == 'random':
		_kernel_knn_cv_median(dataset_all, ds_name, knn_options, mpg_options, kernel_options, mge_options, ged_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary)
	
	elif train_examples == 'best-dataset':
		_kernel_knn_cv_best_ds(dataset_all, ds_name, knn_options, kernel_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary)
		
	elif train_examples == 'trainset':
		_kernel_knn_cv_trainset(dataset_all, ds_name, knn_options, kernel_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary)

	print('\ncomplete.\n')	
	
	
def _kernel_knn_cv_median(dataset_all, ds_name, knn_options, mpg_options, kernel_options, mge_options, ged_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary):
	Gn = dataset_all.graphs
	y_all = dataset_all.targets
	n_neighbors, n_splits, test_size = knn_options['n_neighbors'], knn_options['n_splits'], knn_options['test_size']

	# get shuffles. 
	train_indices, test_indices, train_nums, y_app = _get_shuffles(y_all, n_splits, test_size)
	
	accuracies = [[], [], []]
	for trial in range(len(train_indices)):
		print('\ntrial =', trial)
		
		train_index = train_indices[trial]
		test_index = test_indices[trial]
		G_app = [Gn[i] for i in train_index]
		G_test = [Gn[i] for i in test_index]
		y_test = [y_all[i] for i in test_index]
		gm_unnorm_trial = gram_matrix_unnorm[train_index,:][:,train_index].copy()
		
		# compute pre-images for each class.
		medians = [[], [], []]
		train_nums_tmp = [0] + train_nums
		print('\ncomputing pre-image for each class...\n')
		for i_class in range(len(train_nums_tmp) - 1):
			print(i_class + 1, 'of', len(train_nums_tmp) - 1, 'classes:')
			i_start = int(np.sum(train_nums_tmp[0:i_class + 1]))
			i_end = i_start + train_nums_tmp[i_class + 1]
			median_set = G_app[i_start:i_end]
			
			dataset = dataset_all.copy()
			dataset.load_graphs([g.copy() for g in median_set], targets=None)
			mge_options['update_order'] = True
			mpg_options['gram_matrix_unnorm'] = gm_unnorm_trial[i_start:i_end,i_start:i_end].copy()
			mpg_options['runtime_precompute_gm'] = 0
			set_median, gen_median_uo = _generate_median_preimages(dataset, mpg_options, kernel_options, ged_options, mge_options)
			mge_options['update_order'] = False
			mpg_options['gram_matrix_unnorm'] = gm_unnorm_trial[i_start:i_end,i_start:i_end].copy()
			mpg_options['runtime_precompute_gm'] = 0
			_, gen_median = _generate_median_preimages(dataset, mpg_options, kernel_options, ged_options, mge_options)
			medians[0].append(set_median)
			medians[1].append(gen_median)
			medians[2].append(gen_median_uo)
			
		# for each set of medians.
		print('\nperforming k-nn...')
		for i_app, G_app in enumerate(medians):
			# compute dis_mat between medians.
			dataset = dataset_all.copy()
			dataset.load_graphs([g.copy() for g in G_app], targets=None)
			gm_app_unnorm, _ = _compute_gram_matrix_unnorm(dataset, kernel_options.copy())
			
			# compute the entire Gram matrix.
			graph_kernel = _get_graph_kernel(dataset.copy(), kernel_options.copy())
			kernels_to_medians = []
			for g in G_app:
				kernels_to_median, _ = graph_kernel.compute(g, G_test, **kernel_options.copy()) 
				kernels_to_medians.append(kernels_to_median)
			kernels_to_medians = np.array(kernels_to_medians)
			gm_all = np.concatenate((gm_app_unnorm, kernels_to_medians), axis=1)
			gm_all = np.concatenate((gm_all, np.concatenate((kernels_to_medians.T, gram_matrix_unnorm[test_index,:][:,test_index].copy()), axis=1)), axis=0)
			
			gm_all = normalize_gram_matrix(gm_all.copy())
			dis_mat, _, _, _ = compute_distance_matrix(gm_all)
			
			N = len(G_app)
			
			d_app = dis_mat[range(N),:][:,range(N)].copy()
			
			d_test = np.zeros((N, len(test_index)))
			for i in range(N):
				for j in range(len(test_index)):
					d_test[i, j] = dis_mat[i, j]
					
			accuracies[i_app].append(knn_classification(d_app, d_test, y_app, y_test, n_neighbors, verbose=True, text=train_examples))
			
		# write result detail.
		if save_results:
			f_detail = open(dir_save + fn_output_detail, 'a')
			print('writing results to files...')
			for i, median_type in enumerate(['set-median', 'gen median', 'gen median uo']):
				csv.writer(f_detail).writerow([ds_name, kernel_options['name'], 
					  train_examples + ': ' + median_type, trial, 
					  knn_options['n_neighbors'],
					  len(gm_all), knn_options['test_size'], 
					  accuracies[i][-1][0], accuracies[i][-1][1]])
			f_detail.close()
		
	results = {}
	results['ave_perf_train'] = [np.mean([i[0] for i in j], axis=0) for j in accuracies]
	results['std_perf_train'] = [np.std([i[0] for i in j], axis=0, ddof=1) for j in accuracies]
	results['ave_perf_test'] = [np.mean([i[1] for i in j], axis=0) for j in accuracies]
	results['std_perf_test'] = [np.std([i[1] for i in j], axis=0, ddof=1) for j in accuracies]

	# write result summary for each letter. 
	if save_results:
		f_summary = open(dir_save + fn_output_summary, 'a')
		for i, median_type in enumerate(['set-median', 'gen median', 'gen median uo']):
			csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
				  train_examples + ': ' + median_type, 
				  knn_options['n_neighbors'],
				  knn_options['test_size'], results['ave_perf_train'][i],
				  results['ave_perf_test'][i], results['std_perf_train'][i],
				  results['std_perf_test'][i], time_precompute_gm])
		f_summary.close()
		
		
def _kernel_knn_cv_best_ds(dataset_all, ds_name, knn_options, kernel_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary):
	Gn = dataset_all.graphs
	y_all = dataset_all.targets
	n_neighbors, n_splits, test_size = knn_options['n_neighbors'], knn_options['n_splits'], knn_options['test_size']

	# get shuffles. 
	train_indices, test_indices, train_nums, y_app = _get_shuffles(y_all, n_splits, test_size)
	
	accuracies = []
	for trial in range(len(train_indices)):
		print('\ntrial =', trial)
		
		train_index = train_indices[trial]
		test_index = test_indices[trial]
		G_app = [Gn[i] for i in train_index]
		G_test = [Gn[i] for i in test_index]
		y_test = [y_all[i] for i in test_index]
		gm_unnorm_trial = gram_matrix_unnorm[train_index,:][:,train_index].copy()
		
		# get best graph from trainset according to distance in kernel space for each class.
		best_graphs = []
		train_nums_tmp = [0] + train_nums
		print('\ngetting best graph from trainset for each class...')
		for i_class in range(len(train_nums_tmp) - 1):
			print(i_class + 1, 'of', len(train_nums_tmp) - 1, 'classes.')
			i_start = int(np.sum(train_nums_tmp[0:i_class + 1]))
			i_end = i_start + train_nums_tmp[i_class + 1]
			G_class = G_app[i_start:i_end]
			gm_unnorm_class = gm_unnorm_trial[i_start:i_end,i_start:i_end]
			gm_class = normalize_gram_matrix(gm_unnorm_class.copy())
			
			k_dis_list = []
			for idx in range(len(G_class)):
				k_dis_list.append(compute_k_dis(idx, range(0, len(G_class)), [1 / len(G_class)] * len(G_class), gm_class, withterm3=False))
			idx_k_dis_min = np.argmin(k_dis_list)
			best_graphs.append(G_class[idx_k_dis_min].copy())
			
			
		# perform k-nn.
		print('\nperforming k-nn...')
		# compute dis_mat between medians.
		dataset = dataset_all.copy()
		dataset.load_graphs([g.copy() for g in best_graphs], targets=None)
		gm_app_unnorm, _ = _compute_gram_matrix_unnorm(dataset, kernel_options.copy())
		
		# compute the entire Gram matrix.
		graph_kernel = _get_graph_kernel(dataset.copy(), kernel_options.copy())
		kernels_to_best_graphs = []
		for g in best_graphs:
			kernels_to_best_graph, _ = graph_kernel.compute(g, G_test, **kernel_options.copy()) 
			kernels_to_best_graphs.append(kernels_to_best_graph)
		kernels_to_best_graphs = np.array(kernels_to_best_graphs)
		gm_all = np.concatenate((gm_app_unnorm, kernels_to_best_graphs), axis=1)
		gm_all = np.concatenate((gm_all, np.concatenate((kernels_to_best_graphs.T, gram_matrix_unnorm[test_index,:][:,test_index].copy()), axis=1)), axis=0)
		
		gm_all = normalize_gram_matrix(gm_all.copy())
		dis_mat, _, _, _ = compute_distance_matrix(gm_all)
		
		N = len(best_graphs)
		
		d_app = dis_mat[range(N),:][:,range(N)].copy()
		
		d_test = np.zeros((N, len(test_index)))
		for i in range(N):
			for j in range(len(test_index)):
				d_test[i, j] = dis_mat[i, j]
				
		accuracies.append(knn_classification(d_app, d_test, y_app, y_test, n_neighbors, verbose=True, text=train_examples))
			
		# write result detail.
		if save_results:
			f_detail = open(dir_save + fn_output_detail, 'a')
			print('writing results to files...')
			csv.writer(f_detail).writerow([ds_name, kernel_options['name'], 
				  train_examples, trial, 
				  knn_options['n_neighbors'],
				  len(gm_all), knn_options['test_size'], 
				  accuracies[-1][0], accuracies[-1][1]])
			f_detail.close()
		
	results = {}
	results['ave_perf_train'] = np.mean([i[0] for i in accuracies], axis=0)
	results['std_perf_train'] = np.std([i[0] for i in accuracies], axis=0, ddof=1)
	results['ave_perf_test'] = np.mean([i[1] for i in accuracies], axis=0)
	results['std_perf_test'] = np.std([i[1] for i in accuracies], axis=0, ddof=1)
	
	# write result summary for each letter. 
	if save_results:
		f_summary = open(dir_save + fn_output_summary, 'a')
		csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
				  train_examples, 
				  knn_options['n_neighbors'],
				  knn_options['test_size'], results['ave_perf_train'],
				  results['ave_perf_test'], results['std_perf_train'],
				  results['std_perf_test'], time_precompute_gm])
		f_summary.close()
	
	
def _kernel_knn_cv_trainset(dataset_all, ds_name, knn_options, kernel_options, gram_matrix_unnorm, time_precompute_gm, train_examples, save_results, dir_save, fn_output_detail, fn_output_summary):
	y_all = dataset_all.targets
	n_neighbors, n_splits, test_size = knn_options['n_neighbors'], knn_options['n_splits'], knn_options['test_size']
	
	# compute distance matrix.
	gram_matrix = normalize_gram_matrix(gram_matrix_unnorm.copy())
	dis_mat, _, _, _ = compute_distance_matrix(gram_matrix)

	# get shuffles. 
	train_indices, test_indices, _, _ = _get_shuffles(y_all, n_splits, test_size)
			 
	accuracies = []
	for trial in range(len(train_indices)):
		print('\ntrial =', trial)
		
		train_index = train_indices[trial]
		test_index = test_indices[trial]
		y_app = [y_all[i] for i in train_index]
		y_test = [y_all[i] for i in test_index]
		
		N = len(train_index)
		
		d_app = dis_mat[train_index,:][:,train_index].copy()
		
		d_test = np.zeros((N, len(test_index)))
		for i in range(N):
			for j in range(len(test_index)):
				d_test[i, j] = dis_mat[train_index[i], test_index[j]]
				
		accuracies.append(knn_classification(d_app, d_test, y_app, y_test, n_neighbors, verbose=True, text=train_examples))
		
		# write result detail.
		if save_results:
			print('writing results to files...')
			f_detail = open(dir_save + fn_output_detail, 'a')
			csv.writer(f_detail).writerow([ds_name, kernel_options['name'], 
					  train_examples, trial, knn_options['n_neighbors'],
					  len(gram_matrix), knn_options['test_size'], 
					  accuracies[-1][0], accuracies[-1][1]])
			f_detail.close()
		
	results = {}
	results['ave_perf_train'] = np.mean([i[0] for i in accuracies], axis=0)
	results['std_perf_train'] = np.std([i[0] for i in accuracies], axis=0, ddof=1)
	results['ave_perf_test'] = np.mean([i[1] for i in accuracies], axis=0)
	results['std_perf_test'] = np.std([i[1] for i in accuracies], axis=0, ddof=1)

	# write result summary for each letter. 
	if save_results:
		f_summary = open(dir_save + fn_output_summary, 'a')
		csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
				  train_examples, knn_options['n_neighbors'],
				  knn_options['test_size'], results['ave_perf_train'],
				  results['ave_perf_test'], results['std_perf_train'],
				  results['std_perf_test'], time_precompute_gm])
		f_summary.close()
	
	
def _get_shuffles(y_all, n_splits, test_size):
	rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
	train_indices = [[] for _ in range(n_splits)] 
	test_indices = [[] for _ in range(n_splits)]
	idx_targets = get_same_item_indices(y_all)
	train_nums = []
	keys = []
	for key, item in idx_targets.items():
		i = 0
		for train_i, test_i in rs.split(item): # @todo: careful when parallel.
			train_indices[i] += [item[idx] for idx in train_i]
			test_indices[i] += [item[idx] for idx in test_i]
			i += 1
		train_nums.append(len(train_i))
		keys.append(key)
	return train_indices, test_indices, train_nums, keys
	
	
def _generate_median_preimages(dataset, mpg_options, kernel_options, ged_options, mge_options):
	mpg = MedianPreimageGenerator()
	mpg.dataset = dataset.copy()
	mpg.set_options(**mpg_options.copy())
	mpg.kernel_options = kernel_options.copy()
	mpg.ged_options = ged_options.copy()
	mpg.mge_options = mge_options.copy()
	mpg.run()
	return mpg.set_median, mpg.gen_median


def _get_gram_matrix(load_gm, dir_save, ds_name, kernel_options, dataset_all):
	if load_gm == 'auto':
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile_exist = os.path.isfile(os.path.abspath(gm_fname))
		if gmfile_exist:
			gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
			gram_matrix_unnorm = gmfile['gram_matrix_unnorm']
			time_precompute_gm = float(gmfile['run_time'])
		else:
			gram_matrix_unnorm, time_precompute_gm = _compute_gram_matrix_unnorm(dataset_all, kernel_options)
			np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm=gram_matrix_unnorm, run_time=time_precompute_gm)
	elif not load_gm:
		gram_matrix_unnorm, time_precompute_gm = _compute_gram_matrix_unnorm(dataset_all, kernel_options)
		np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm=gram_matrix_unnorm, run_time=time_precompute_gm)
	else:
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile = np.load(gm_fname, allow_pickle=True)
		gram_matrix_unnorm = gmfile['gram_matrix_unnorm']
		time_precompute_gm = float(gmfile['run_time'])
		
	return gram_matrix_unnorm, time_precompute_gm


def _get_graph_kernel(dataset, kernel_options):
	from gklearn.utils.utils import get_graph_kernel_by_name
	graph_kernel = get_graph_kernel_by_name(kernel_options['name'], 
				  node_labels=dataset.node_labels,
				  edge_labels=dataset.edge_labels, 
				  node_attrs=dataset.node_attrs,
				  edge_attrs=dataset.edge_attrs,
				  ds_infos=dataset.get_dataset_infos(keys=['directed']),
				  kernel_options=kernel_options)
	return graph_kernel
		
		
def _compute_gram_matrix_unnorm(dataset, kernel_options):
	from gklearn.utils.utils import get_graph_kernel_by_name
	graph_kernel = get_graph_kernel_by_name(kernel_options['name'], 
				  node_labels=dataset.node_labels,
				  edge_labels=dataset.edge_labels, 
				  node_attrs=dataset.node_attrs,
				  edge_attrs=dataset.edge_attrs,
				  ds_infos=dataset.get_dataset_infos(keys=['directed']),
				  kernel_options=kernel_options)
	
	gram_matrix, run_time = graph_kernel.compute(dataset.graphs, **kernel_options)
	gram_matrix_unnorm = graph_kernel.gram_matrix_unnorm
		
	return gram_matrix_unnorm, run_time
		
		
def _init_output_file_knn(ds_name, gkernel, fit_method, dir_output):
	if not os.path.exists(dir_output):
		os.makedirs(dir_output)
	fn_output_detail = 'results_detail_knn.' + ds_name + '.' + gkernel + '.csv'
	f_detail = open(dir_output + fn_output_detail, 'a')
	csv.writer(f_detail).writerow(['dataset', 'graph kernel',
			  'train examples', 'trial', 'num neighbors', 'num graphs', 'test size',
			  'perf train', 'perf test'])
	f_detail.close()
	
	fn_output_summary = 'results_summary_knn.' + ds_name + '.' + gkernel + '.csv'
	f_summary = open(dir_output + fn_output_summary, 'a')
	csv.writer(f_summary).writerow(['dataset', 'graph kernel',
		      'train examples', 'num neighbors', 'test size',
			  'ave perf train', 'ave perf test',
			  'std perf train', 'std perf test', 'time precompute gm'])
	f_summary.close()
	
	return fn_output_detail, fn_output_summary