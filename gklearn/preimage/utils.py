#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:05:07 2019

Useful functions.
@author: ljia
"""
#import networkx as nx

import multiprocessing
import numpy as np

from gklearn.kernels.marginalizedKernel import marginalizedkernel
from gklearn.kernels.untilHPathKernel import untilhpathkernel
from gklearn.kernels.spKernel import spkernel
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct, polynomialkernel
from gklearn.kernels.structuralspKernel import structuralspkernel
from gklearn.kernels.treeletKernel import treeletkernel
from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
from gklearn.utils import Dataset
import csv
import networkx as nx
import os


def generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=True, save_medians=True, plot_medians=True, load_gm='auto', dir_save='', irrelevant_labels=None, edge_required=False, cut_range=None):
	import os.path
	from gklearn.preimage import MedianPreimageGenerator
	from gklearn.utils import split_dataset_by_target
	from gklearn.utils.graphfiles import saveGXL
	
	# 1. get dataset.
	print('1. getting dataset...')
	dataset_all = Dataset()
	dataset_all.load_predefined_dataset(ds_name)
	dataset_all.trim_dataset(edge_required=edge_required)
	if irrelevant_labels is not None:
		dataset_all.remove_labels(**irrelevant_labels)
	if cut_range is not None:
		dataset_all.cut_graphs(cut_range)
	datasets = split_dataset_by_target(dataset_all)

	if save_results:
		# create result files.
		print('creating output files...')
		fn_output_detail, fn_output_summary = _init_output_file_preimage(ds_name, kernel_options['name'], mpg_options['fit_method'], dir_save)
		
	sod_sm_list = []
	sod_gm_list = []
	dis_k_sm_list = []
	dis_k_gm_list = []
	dis_k_gi_min_list = []
	time_optimize_ec_list = []
	time_generate_list = []
	time_total_list = []
	itrs_list = []
	converged_list = []
	num_updates_ecc_list = []
	mge_decrease_order_list = []
	mge_increase_order_list = []
	mge_converged_order_list = []
	nb_sod_sm2gm = [0, 0, 0]
	nb_dis_k_sm2gm = [0, 0, 0]
	nb_dis_k_gi2sm = [0, 0, 0]
	nb_dis_k_gi2gm = [0, 0, 0]
	dis_k_max_list = []	
	dis_k_min_list = []
	dis_k_mean_list = []
	if load_gm == 'auto':
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile_exist = os.path.isfile(os.path.abspath(gm_fname))
		if gmfile_exist:
			gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
			gram_matrix_unnorm_list = [item for item in gmfile['gram_matrix_unnorm_list']]
			time_precompute_gm_list = gmfile['run_time_list'].tolist()
		else:
			gram_matrix_unnorm_list = []
			time_precompute_gm_list = []
	elif not load_gm:
		gram_matrix_unnorm_list = []
		time_precompute_gm_list = []
	else:
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
		gram_matrix_unnorm_list = [item for item in gmfile['gram_matrix_unnorm_list']]
		time_precompute_gm_list = gmfile['run_time_list'].tolist()
#	repeats_better_sod_sm2gm = []
#	repeats_better_dis_k_sm2gm = []
#	repeats_better_dis_k_gi2sm = []
#	repeats_better_dis_k_gi2gm = []		
		
	print('starting generating preimage for each class of target...')
	idx_offset = 0
	for idx, dataset in enumerate(datasets):
		target = dataset.targets[0]
		print('\ntarget =', target, '\n')
#		if target != 1:
# 			continue
		
		num_graphs = len(dataset.graphs)
		if num_graphs < 2:
			print('\nnumber of graphs = ', num_graphs, ', skip.\n')
			idx_offset += 1
			continue
			
		# 2. set parameters.
		print('2. initializing mpg and setting parameters...')
		if load_gm:
			if gmfile_exist:
				mpg_options['gram_matrix_unnorm'] = gram_matrix_unnorm_list[idx - idx_offset]
				mpg_options['runtime_precompute_gm'] = time_precompute_gm_list[idx - idx_offset]
		mpg = MedianPreimageGenerator()
		mpg.dataset = dataset
		mpg.set_options(**mpg_options.copy())
		mpg.kernel_options = kernel_options.copy()
		mpg.ged_options = ged_options.copy()
		mpg.mge_options = mge_options.copy()

		# 3. compute median preimage.
		print('3. computing median preimage...')
		mpg.run()
		results = mpg.get_results()
		
		# 4. compute pairwise kernel distances.
		print('4. computing pairwise kernel distances...')
		_, dis_k_max, dis_k_min, dis_k_mean = mpg.graph_kernel.compute_distance_matrix()
		dis_k_max_list.append(dis_k_max)
		dis_k_min_list.append(dis_k_min)
		dis_k_mean_list.append(dis_k_mean)
		
		# 5. save results (and median graphs).
		print('5. saving results (and median graphs)...')
		# write result detail.
		if save_results:
			print('writing results to files...')
			sod_sm2gm = get_relations(np.sign(results['sod_gen_median'] - results['sod_set_median']))
			dis_k_sm2gm = get_relations(np.sign(results['k_dis_gen_median'] - results['k_dis_set_median']))
			dis_k_gi2sm = get_relations(np.sign(results['k_dis_set_median'] - results['k_dis_dataset']))
			dis_k_gi2gm = get_relations(np.sign(results['k_dis_gen_median'] - results['k_dis_dataset']))

			f_detail = open(dir_save + fn_output_detail, 'a')
			csv.writer(f_detail).writerow([ds_name, kernel_options['name'], 
					  ged_options['edit_cost'], ged_options['method'], 
					  ged_options['attr_distance'], mpg_options['fit_method'], 
					  num_graphs, target, 1, 
					  results['sod_set_median'], results['sod_gen_median'],
					  results['k_dis_set_median'], results['k_dis_gen_median'], 
					  results['k_dis_dataset'], sod_sm2gm, dis_k_sm2gm, 
					  dis_k_gi2sm, dis_k_gi2gm,	results['edit_cost_constants'],
					  results['runtime_precompute_gm'], results['runtime_optimize_ec'], 
					  results['runtime_generate_preimage'], results['runtime_total'],
					  results['itrs'], results['converged'],
					  results['num_updates_ecc'],
					  results['mge']['num_decrease_order'] > 0, # @todo: not suitable for multi-start mge
					  results['mge']['num_increase_order'] > 0,
					  results['mge']['num_converged_descents'] > 0])
			f_detail.close()
		
			# compute result summary.
			sod_sm_list.append(results['sod_set_median'])
			sod_gm_list.append(results['sod_gen_median'])
			dis_k_sm_list.append(results['k_dis_set_median'])
			dis_k_gm_list.append(results['k_dis_gen_median'])
			dis_k_gi_min_list.append(results['k_dis_dataset'])
			time_precompute_gm_list.append(results['runtime_precompute_gm'])
			time_optimize_ec_list.append(results['runtime_optimize_ec'])
			time_generate_list.append(results['runtime_generate_preimage'])
			time_total_list.append(results['runtime_total'])
			itrs_list.append(results['itrs'])
			converged_list.append(results['converged'])
			num_updates_ecc_list.append(results['num_updates_ecc'])
			mge_decrease_order_list.append(results['mge']['num_decrease_order'] > 0)
			mge_increase_order_list.append(results['mge']['num_increase_order'] > 0)
			mge_converged_order_list.append(results['mge']['num_converged_descents'] > 0)
			# # SOD SM -> GM
			if results['sod_set_median'] > results['sod_gen_median']:
				nb_sod_sm2gm[0] += 1
	#			repeats_better_sod_sm2gm.append(1)
			elif results['sod_set_median'] == results['sod_gen_median']:
				nb_sod_sm2gm[1] += 1
			elif results['sod_set_median'] < results['sod_gen_median']:
				nb_sod_sm2gm[2] += 1
			# # dis_k SM -> GM
			if results['k_dis_set_median'] > results['k_dis_gen_median']:
				nb_dis_k_sm2gm[0] += 1
	#			repeats_better_dis_k_sm2gm.append(1)
			elif results['k_dis_set_median'] == results['k_dis_gen_median']:
				nb_dis_k_sm2gm[1] += 1
			elif results['k_dis_set_median'] < results['k_dis_gen_median']:
				nb_dis_k_sm2gm[2] += 1
			# # dis_k gi -> SM
			if results['k_dis_dataset'] > results['k_dis_set_median']:
				nb_dis_k_gi2sm[0] += 1
	#			repeats_better_dis_k_gi2sm.append(1)
			elif results['k_dis_dataset'] == results['k_dis_set_median']:
				nb_dis_k_gi2sm[1] += 1
			elif results['k_dis_dataset'] < results['k_dis_set_median']:
				nb_dis_k_gi2sm[2] += 1
			# # dis_k gi -> GM
			if results['k_dis_dataset'] > results['k_dis_gen_median']:
				nb_dis_k_gi2gm[0] += 1
	#			repeats_better_dis_k_gi2gm.append(1)
			elif results['k_dis_dataset'] == results['k_dis_gen_median']:
				nb_dis_k_gi2gm[1] += 1
			elif results['k_dis_dataset'] < results['k_dis_gen_median']:
				nb_dis_k_gi2gm[2] += 1

			# write result summary for each letter.
			f_summary = open(dir_save + fn_output_summary, 'a')
			csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
					  ged_options['edit_cost'], ged_options['method'], 
					  ged_options['attr_distance'], mpg_options['fit_method'], 
					  num_graphs, target,
					  results['sod_set_median'], results['sod_gen_median'],
					  results['k_dis_set_median'], results['k_dis_gen_median'], 
					  results['k_dis_dataset'], sod_sm2gm, dis_k_sm2gm, 
					  dis_k_gi2sm, dis_k_gi2gm, 
					  results['runtime_precompute_gm'], results['runtime_optimize_ec'], 
					  results['runtime_generate_preimage'], results['runtime_total'],
					  results['itrs'], results['converged'],
					  results['num_updates_ecc'], 
					  results['mge']['num_decrease_order'] > 0, # @todo: not suitable for multi-start mge
					  results['mge']['num_increase_order'] > 0,
					  results['mge']['num_converged_descents'] > 0, 
					  nb_sod_sm2gm, 
					  nb_dis_k_sm2gm, nb_dis_k_gi2sm, nb_dis_k_gi2gm])
			f_summary.close()
			 
		# save median graphs.
		if save_medians:
			os.makedirs(dir_save + 'medians/', exist_ok=True)
			print('Saving median graphs to files...')
			fn_pre_sm = dir_save + 'medians/set_median.' + mpg_options['fit_method'] + '.nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(mpg.set_median, fn_pre_sm + '.gxl', method='default', 
					node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
			fn_pre_gm = dir_save + 'medians/gen_median.' + mpg_options['fit_method'] + '.nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(mpg.gen_median, fn_pre_gm + '.gxl', method='default',
		  			node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
			fn_best_dataset = dir_save + 'medians/g_best_dataset.' + mpg_options['fit_method'] + '.nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(mpg.best_from_dataset, fn_best_dataset + '.gxl', method='default',
		  			node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
		
		# plot median graphs.
		if plot_medians and save_medians:
			if ged_options['edit_cost'] == 'LETTER2' or ged_options['edit_cost'] == 'LETTER' or ds_name == 'Letter-high' or ds_name == 'Letter-med' or ds_name == 'Letter-low':			
				draw_Letter_graph(mpg.set_median, fn_pre_sm)
				draw_Letter_graph(mpg.gen_median, fn_pre_gm)
				draw_Letter_graph(mpg.best_from_dataset, fn_best_dataset)
				
		if (load_gm == 'auto' and not gmfile_exist) or not load_gm:
			gram_matrix_unnorm_list.append(mpg.gram_matrix_unnorm)

	# write result summary for each class. 
	if save_results:
		sod_sm_mean = np.mean(sod_sm_list)
		sod_gm_mean = np.mean(sod_gm_list)
		dis_k_sm_mean = np.mean(dis_k_sm_list)
		dis_k_gm_mean = np.mean(dis_k_gm_list)
		dis_k_gi_min_mean = np.mean(dis_k_gi_min_list)
		time_precompute_gm_mean = np.mean(time_precompute_gm_list)
		time_optimize_ec_mean = np.mean(time_optimize_ec_list)
		time_generate_mean = np.mean(time_generate_list)
		time_total_mean = np.mean(time_total_list)
		itrs_mean = np.mean(itrs_list)
		num_converged = np.sum(converged_list)
		num_updates_ecc_mean = np.mean(num_updates_ecc_list)
		num_mge_decrease_order = np.sum(mge_decrease_order_list)
		num_mge_increase_order = np.sum(mge_increase_order_list)
		num_mge_converged = np.sum(mge_converged_order_list)
		sod_sm2gm_mean = get_relations(np.sign(sod_gm_mean - sod_sm_mean))
		dis_k_sm2gm_mean = get_relations(np.sign(dis_k_gm_mean - dis_k_sm_mean))
		dis_k_gi2sm_mean = get_relations(np.sign(dis_k_sm_mean - dis_k_gi_min_mean))
		dis_k_gi2gm_mean = get_relations(np.sign(dis_k_gm_mean - dis_k_gi_min_mean))
		f_summary = open(dir_save + fn_output_summary, 'a')
		csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
				  ged_options['edit_cost'], ged_options['method'], 
				  ged_options['attr_distance'], mpg_options['fit_method'], 
				  num_graphs, 'all',
				  sod_sm_mean, sod_gm_mean, dis_k_sm_mean, dis_k_gm_mean,
				  dis_k_gi_min_mean, sod_sm2gm_mean, dis_k_sm2gm_mean, 
				  dis_k_gi2sm_mean, dis_k_gi2gm_mean,
				  time_precompute_gm_mean, time_optimize_ec_mean,
				  time_generate_mean, time_total_mean, itrs_mean, 
				  num_converged, num_updates_ecc_mean,
				  num_mge_decrease_order, num_mge_increase_order,
				  num_mge_converged])
		f_summary.close()
		
	# save total pairwise kernel distances.
	dis_k_max = np.max(dis_k_max_list)
	dis_k_min = np.min(dis_k_min_list)
	dis_k_mean = np.mean(dis_k_mean_list)
	print('The maximum pairwise distance in kernel space:', dis_k_max)
	print('The minimum pairwise distance in kernel space:', dis_k_min)
	print('The average pairwise distance in kernel space:', dis_k_mean)
	
	# write Gram matrices to file.
	if (load_gm == 'auto' and not gmfile_exist) or not load_gm:
		np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm_list=gram_matrix_unnorm_list, run_time_list=time_precompute_gm_list)	

	print('\ncomplete.\n')	

	
def _init_output_file_preimage(ds_name, gkernel, fit_method, dir_output):
	os.makedirs(dir_output, exist_ok=True)
#	fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.' + fit_method + '.csv'
	fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.csv'
	f_detail = open(dir_output + fn_output_detail, 'a')
	csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'edit cost', 
			  'GED method', 'attr distance', 'fit method', 'num graphs', 
			  'target', 'repeat', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
			  'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
			  'dis_k gi -> GM', 'edit cost constants', 'time precompute gm',
			  'time optimize ec', 'time generate preimage', 'time total',
			  'itrs', 'converged', 'num updates ecc', 'mge decrease order', 
			  'mge increase order', 'mge converged'])
	f_detail.close()
	
#	fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.' + fit_method + '.csv'
	fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.csv'
	f_summary = open(dir_output + fn_output_summary, 'a')
	csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'edit cost', 
			  'GED method', 'attr distance', 'fit method', 'num graphs', 
			  'target', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
			  'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
			  'dis_k gi -> GM', 'time precompute gm', 'time optimize ec', 
			  'time generate preimage', 'time total', 'itrs', 'num converged', 
			  'num updates ecc', 'mge num decrease order', 'mge num increase order', 
			  'mge num converged', '# SOD SM -> GM', '# dis_k SM -> GM', 
			  '# dis_k gi -> SM', '# dis_k gi -> GM'])
#			   'repeats better SOD SM -> GM', 
#			  'repeats better dis_k SM -> GM', 'repeats better dis_k gi -> SM', 
#			  'repeats better dis_k gi -> GM'])
	f_summary.close()
	
	return fn_output_detail, fn_output_summary


def get_relations(sign):
	if sign == -1:
		return 'better'
	elif sign == 0:
		return 'same'
	elif sign == 1:
		return 'worse'
	
	
#Dessin median courrant
def draw_Letter_graph(graph, file_prefix):
	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pyplot as plt
	plt.figure()
	pos = {}
	for n in graph.nodes:
		pos[n] = np.array([float(graph.nodes[n]['x']),float(graph.nodes[n]['y'])])
	nx.draw_networkx(graph, pos)
	plt.savefig(file_prefix + '.eps', format='eps', dpi=300)
#	plt.show()
	plt.clf()
	plt.close()


def remove_edges(Gn):
	for G in Gn:
		for _, _, attrs in G.edges(data=True):
			attrs.clear()
			
			
def dis_gstar(idx_g, idx_gi, alpha, Kmatrix, term3=0, withterm3=True):
	term1 = Kmatrix[idx_g, idx_g]
	term2 = 0
	for i, a in enumerate(alpha):
		term2 += a * Kmatrix[idx_g, idx_gi[i]]
	term2 *= 2
	if withterm3 == False:
		for i1, a1 in enumerate(alpha):
			for i2, a2 in enumerate(alpha):
				term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
	return np.sqrt(term1 - term2 + term3)


def compute_k_dis(idx_g, idx_gi, alphas, Kmatrix, term3=0, withterm3=True):
	term1 = Kmatrix[idx_g, idx_g]
	term2 = 0
	for i, a in enumerate(alphas):
		term2 += a * Kmatrix[idx_g, idx_gi[i]]
	term2 *= 2
	if withterm3 == False:
		for i1, a1 in enumerate(alphas):
			for i2, a2 in enumerate(alphas):
				term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
	return np.sqrt(term1 - term2 + term3)


def compute_kernel(Gn, graph_kernel, node_label, edge_label, verbose, parallel='imap_unordered'):
	if graph_kernel == 'marginalizedkernel':
		Kmatrix, _ = marginalizedkernel(Gn, node_label=node_label, edge_label=edge_label,
								  p_quit=0.03, n_iteration=10, remove_totters=False,
								  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
	elif graph_kernel == 'untilhpathkernel':
		Kmatrix, _ = untilhpathkernel(Gn, node_label=node_label, edge_label=edge_label,
								  depth=7, k_func='MinMax', compute_method='trie',
								  parallel=parallel,
								  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
	elif graph_kernel == 'spkernel':
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		Kmatrix = np.empty((len(Gn), len(Gn)))
#		Kmatrix[:] = np.nan
		Kmatrix, _, idx = spkernel(Gn, node_label=node_label, node_kernels=
							  {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
							  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
#		for i, row in enumerate(idx):
#			for j, col in enumerate(idx):
#				Kmatrix[row, col] = Kmatrix_tmp[i, j]
	elif graph_kernel == 'structuralspkernel':
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		Kmatrix, _ = structuralspkernel(Gn, node_label=node_label, 
							  edge_label=edge_label, node_kernels=sub_kernels,
							  edge_kernels=sub_kernels,
							  parallel=parallel, n_jobs=multiprocessing.cpu_count(), 
							  verbose=verbose)
	elif graph_kernel == 'treeletkernel':
		pkernel = functools.partial(polynomialkernel, d=2, c=1e5)
#		pkernel = functools.partial(gaussiankernel, gamma=1e-6)
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		Kmatrix, _ = treeletkernel(Gn, node_label=node_label, edge_label=edge_label,
								   sub_kernel=pkernel, parallel=parallel,
								   n_jobs=multiprocessing.cpu_count(), verbose=verbose)
	elif graph_kernel == 'weisfeilerlehmankernel':
		Kmatrix, _ = weisfeilerlehmankernel(Gn, node_label=node_label, edge_label=edge_label,
								   height=4, base_kernel='subtree', parallel=None,
								   n_jobs=multiprocessing.cpu_count(), verbose=verbose)
	else:
		raise Exception('The graph kernel "', graph_kernel, '" is not defined.')	
		
	# normalization
	Kmatrix_diag = Kmatrix.diagonal().copy()
	for i in range(len(Kmatrix)):
		for j in range(i, len(Kmatrix)):
			Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
			Kmatrix[j][i] = Kmatrix[i][j]
	return Kmatrix
			

def gram2distances(Kmatrix):
	dmatrix = np.zeros((len(Kmatrix), len(Kmatrix)))
	for i1 in range(len(Kmatrix)):
		for i2 in range(len(Kmatrix)):
			dmatrix[i1, i2] = Kmatrix[i1, i1] + Kmatrix[i2, i2] - 2 * Kmatrix[i1, i2]
	dmatrix = np.sqrt(dmatrix)
	return dmatrix


def kernel_distance_matrix(Gn, node_label, edge_label, Kmatrix=None, 
						   gkernel=None, verbose=True):
	import warnings
	warnings.warn('gklearn.preimage.utils.kernel_distance_matrix is deprecated, use gklearn.kernels.graph_kernel.compute_distance_matrix or gklearn.utils.compute_distance_matrix instead', DeprecationWarning)
	dis_mat = np.empty((len(Gn), len(Gn)))
	if Kmatrix is None:
		Kmatrix = compute_kernel(Gn, gkernel, node_label, edge_label, verbose)
	for i in range(len(Gn)):
		for j in range(i, len(Gn)):
			dis = Kmatrix[i, i] + Kmatrix[j, j] - 2 * Kmatrix[i, j]
			if dis < 0:
				if dis > -1e-10:
					dis = 0
				else:
					raise ValueError('The distance is negative.')
			dis_mat[i, j] = np.sqrt(dis)
			dis_mat[j, i] = dis_mat[i, j]
	dis_max = np.max(np.max(dis_mat))
	dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
	dis_mean = np.mean(np.mean(dis_mat))
	return dis_mat, dis_max, dis_min, dis_mean


def get_same_item_indices(ls):
	"""Get the indices of the same items in a list. Return a dict keyed by items.
	"""
	idx_dict = {}
	for idx, item in enumerate(ls):
		if item in idx_dict:
			idx_dict[item].append(idx)
		else:
			idx_dict[item] = [idx]
	return idx_dict


def k_nearest_neighbors_to_median_in_kernel_space(Gn, Kmatrix=None, gkernel=None,
												  node_label=None, edge_label=None):
	dis_k_all = [] # distance between g_star and each graph.
	alpha = [1 / len(Gn)] * len(Gn)
	if Kmatrix is None:
		Kmatrix = compute_kernel(Gn, gkernel, node_label, edge_label, True)
	term3 = 0
	for i1, a1 in enumerate(alpha):
		for i2, a2 in enumerate(alpha):
			term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
	for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
		dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
		dis_all.append(dtemp)


def normalize_distance_matrix(D):
	max_value = np.amax(D)
	min_value = np.amin(D)
	return (D - min_value) / (max_value - min_value)