#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:03:01 2020

@author: ljia
"""
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from gklearn.utils.utils import get_graph_kernel_by_name
# from gklearn.preimage.utils import get_same_item_indices

def sum_squares(a, b):
	"""
	Return the sum of squares of the difference between a and b, aka MSE
	"""
	return np.sum([(a[i] - b[i])**2 for i in range(len(a))])


def euclid_d(x, y):
	"""
	1D euclidean distance
	"""
	return np.sqrt((x-y)**2)


def man_d(x, y):
	"""
	1D manhattan distance
	"""
	return np.abs((x-y))


def knn_regression(D_app, D_test, y_app, y_test, n_neighbors, verbose=True, text=None):

	from sklearn.neighbors import KNeighborsRegressor
	knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='precomputed')
	knn.fit(D_app, y_app)
	y_pred = knn.predict(D_app)
	y_pred_test = knn.predict(D_test.T)
	perf_app = np.sqrt(sum_squares(y_pred, y_app)/len(y_app))
	perf_test = np.sqrt(sum_squares(y_pred_test, y_test)/len(y_test))

	if (verbose):
		print("Learning error with {} train examples : {}".format(text, perf_app))
		print("Test error with {} train examples : {}".format(text, perf_test))

	return perf_app, perf_test


def knn_classification(d_app, d_test, y_app, y_test, n_neighbors, verbose=True, text=None):
	knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
	knn.fit(d_app, y_app)
	y_pred = knn.predict(d_app)
	y_pred_test = knn.predict(d_test.T)
	perf_app = accuracy_score(y_app, y_pred)
	perf_test = accuracy_score(y_test, y_pred_test)

	if (verbose):
		print("Learning accuracy with {} costs : {}".format(text, perf_app))
		print("Test accuracy with {} costs : {}".format(text, perf_test))
		
	return perf_app, perf_test
	

def knn_cv(dataset, kernel_options, trainset=None, n_neighbors=1, n_splits=50, test_size=0.9, verbose=True):
	'''
	Perform a knn classification cross-validation on given dataset.
	'''
# 	Gn = dataset.graphs
	y_all = dataset.targets
	
	# compute kernel distances.
	dis_mat = _compute_kernel_distances(dataset, kernel_options, trainset=trainset)
		
	
	rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
# 	train_indices = [[] for _ in range(n_splits)] 
# 	test_indices = [[] for _ in range(n_splits)]
# 	idx_targets = get_same_item_indices(y_all)
# 	for key, item in idx_targets.items():
# 		i = 0
# 		for train_i, test_i in rs.split(item): # @todo: careful when parallel.
# 			train_indices[i] += [item[idx] for idx in train_i]
# 			test_indices[i] += [item[idx] for idx in test_i]
# 			i += 1
	
	accuracies = []
# 	for trial in range(len(train_indices)):
# 		train_index = train_indices[trial]
# 		test_index = test_indices[trial] 
	for train_index, test_index in rs.split(y_all):
# 		print(train_index, test_index)
# 		G_app = [Gn[i] for i in train_index]
# 		G_test = [Gn[i] for i in test_index]
		y_app = [y_all[i] for i in train_index]
		y_test = [y_all[i] for i in test_index]
		
		N = len(train_index)
		
		d_app = dis_mat.copy()
		d_app = d_app[train_index,:]
		d_app = d_app[:,train_index]
		
		d_test = np.zeros((N, len(test_index)))
		
		for i in range(N):
			for j in range(len(test_index)):
				d_test[i, j] = dis_mat[train_index[i], test_index[j]]
				
		accuracies.append(knn_classification(d_app, d_test, y_app, y_test, n_neighbors, verbose=verbose, text=''))
		
	results = {}
	results['ave_perf_train'] = np.mean([i[0] for i in accuracies], axis=0)
	results['std_perf_train'] = np.std([i[0] for i in accuracies], axis=0, ddof=1)
	results['ave_perf_test'] = np.mean([i[1] for i in accuracies], axis=0)
	results['std_perf_test'] = np.std([i[1] for i in accuracies], axis=0, ddof=1)

	return results
		
		
def _compute_kernel_distances(dataset, kernel_options, trainset=None):
	graph_kernel = get_graph_kernel_by_name(kernel_options['name'], 
				  node_labels=dataset.node_labels,
				  edge_labels=dataset.edge_labels, 
				  node_attrs=dataset.node_attrs,
				  edge_attrs=dataset.edge_attrs,
				  ds_infos=dataset.get_dataset_infos(keys=['directed']),
				  kernel_options=kernel_options)
	
	gram_matrix, run_time = graph_kernel.compute(dataset.graphs, **kernel_options)

	dis_mat, _, _, _ = graph_kernel.compute_distance_matrix()
	
	if trainset is not None:
		gram_matrix_unnorm = graph_kernel.gram_matrix_unnorm
		

	return dis_mat