#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen May 27 14:27:15 2020

@author: ljia
"""
import numpy as np
import csv
import os
import os.path
from gklearn.utils import Dataset
from gklearn.preimage import MedianPreimageGenerator
from gklearn.utils import normalize_gram_matrix
from gklearn.utils import split_dataset_by_target
from gklearn.preimage.utils import compute_k_dis
from gklearn.utils.graphfiles import saveGXL
import networkx as nx
	

def remove_best_graph(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=True, save_medians=True, plot_medians=True, load_gm='auto', dir_save='', irrelevant_labels=None, edge_required=False, cut_range=None):
	"""Remove the best graph from the median set w.r.t. distance in kernel space, and to see if it is possible to generate the removed graph using the graphs left in the median set.
	"""
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
		fn_output_detail, fn_output_summary = _init_output_file(ds_name, kernel_options['name'], mpg_options['fit_method'], dir_save)
	else:
		fn_output_detail, fn_output_summary = None, None
		
	# 2. compute/load Gram matrix a priori.
	print('2. computing/loading Gram matrix...')
	gram_matrix_unnorm_list, time_precompute_gm_list = _get_gram_matrix(load_gm, dir_save, ds_name, kernel_options, datasets)
		
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
	best_dis_list = []
	print('starting experiment for each class of target...')
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

		# 3. get the best graph and remove it from median set.
		print('3. getting and removing the best graph...')
		gram_matrix_unnorm = gram_matrix_unnorm_list[idx - idx_offset]
		best_index, best_dis, best_graph = _get_best_graph([g.copy() for g in dataset.graphs], normalize_gram_matrix(gram_matrix_unnorm.copy()))
		median_set_new = [dataset.graphs[i] for i in range(len(dataset.graphs)) if i != best_index]
		num_graphs -= 1
		if num_graphs == 1:
			continue		
		best_dis_list.append(best_dis)
		
		dataset.load_graphs(median_set_new, targets=None)
		gram_matrix_unnorm_new = np.delete(gram_matrix_unnorm, best_index, axis=0)
		gram_matrix_unnorm_new = np.delete(gram_matrix_unnorm_new, best_index, axis=1)
			
		# 4. set parameters.
		print('4. initializing mpg and setting parameters...')	
		mpg_options['gram_matrix_unnorm'] = gram_matrix_unnorm_new
		mpg_options['runtime_precompute_gm'] = time_precompute_gm_list[idx - idx_offset]
		mpg = MedianPreimageGenerator()
		mpg.dataset = dataset
		mpg.set_options(**mpg_options.copy())
		mpg.kernel_options = kernel_options.copy()
		mpg.ged_options = ged_options.copy()
		mpg.mge_options = mge_options.copy()

		# 5. compute median preimage.
		print('5. computing median preimage...')
		mpg.run()
		results = mpg.get_results()
		
		# 6. compute pairwise kernel distances.
		print('6. computing pairwise kernel distances...')
		_, dis_k_max, dis_k_min, dis_k_mean = mpg.graph_kernel.compute_distance_matrix()
		dis_k_max_list.append(dis_k_max)
		dis_k_min_list.append(dis_k_min)
		dis_k_mean_list.append(dis_k_mean)
		
		# 7. save results (and median graphs).
		print('7. saving results (and median graphs)...')
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
					  results['k_dis_dataset'], best_dis, best_index,
					  sod_sm2gm, dis_k_sm2gm, 
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
					  results['k_dis_dataset'], best_dis, best_index,
					  sod_sm2gm, dis_k_sm2gm, 
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
			saveGXL(best_graph, fn_best_dataset + '.gxl', method='default',
		  			node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
			fn_best_median_set = dir_save + 'medians/g_best_median_set.' + mpg_options['fit_method'] + '.nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(mpg.best_from_dataset, fn_best_median_set + '.gxl', method='default',
		  			node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
		
		# plot median graphs.
		if plot_medians and save_medians:
			if ged_options['edit_cost'] == 'LETTER2' or ged_options['edit_cost'] == 'LETTER' or ds_name == 'Letter-high' or ds_name == 'Letter-med' or ds_name == 'Letter-low':			
				draw_Letter_graph(mpg.set_median, fn_pre_sm)
				draw_Letter_graph(mpg.gen_median, fn_pre_gm)
				draw_Letter_graph(mpg.best_from_dataset, fn_best_dataset)

	# write result summary for each letter. 
	if save_results:
		sod_sm_mean = np.mean(sod_sm_list)
		sod_gm_mean = np.mean(sod_gm_list)
		dis_k_sm_mean = np.mean(dis_k_sm_list)
		dis_k_gm_mean = np.mean(dis_k_gm_list)
		dis_k_gi_min_mean = np.mean(dis_k_gi_min_list)
		best_dis_mean = np.mean(best_dis_list)
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
				  dis_k_gi_min_mean, best_dis_mean, '-', 
				  sod_sm2gm_mean, dis_k_sm2gm_mean, 
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

	print('\ncomplete.\n')	


def _get_best_graph(Gn, gram_matrix):
	k_dis_list = []
	for idx in range(len(Gn)):
		k_dis_list.append(compute_k_dis(idx, range(0, len(Gn)), [1 / len(Gn)] * len(Gn), gram_matrix, withterm3=False))
	best_index = np.argmin(k_dis_list)
	best_dis = k_dis_list[best_index]
	best_graph = Gn[best_index].copy()
	return best_index, best_dis, best_graph 


def get_relations(sign):
	if sign == -1:
		return 'better'
	elif sign == 0:
		return 'same'
	elif sign == 1:
		return 'worse'


def _get_gram_matrix(load_gm, dir_save, ds_name, kernel_options, datasets):
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
			for dataset in datasets:
				gram_matrix_unnorm, time_precompute_gm = _compute_gram_matrix_unnorm(dataset, kernel_options)
				gram_matrix_unnorm_list.append(gram_matrix_unnorm)
				time_precompute_gm_list.append(time_precompute_gm)
			np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm_list=gram_matrix_unnorm_list, run_time_list=time_precompute_gm_list)
	elif not load_gm:
		gram_matrix_unnorm_list = []
		time_precompute_gm_list = []
		for dataset in datasets:
			gram_matrix_unnorm, time_precompute_gm = _compute_gram_matrix_unnorm(dataset, kernel_options)
			gram_matrix_unnorm_list.append(gram_matrix_unnorm)
			time_precompute_gm_list.append(time_precompute_gm)
		np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm_list=gram_matrix_unnorm_list, run_time_list=time_precompute_gm_list)
	else:
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
		gram_matrix_unnorm_list = [item for item in gmfile['gram_matrix_unnorm_list']]
		time_precompute_gm_list = gmfile['run_time_list'].tolist()
		
	return gram_matrix_unnorm_list, time_precompute_gm_list


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
		
		
def _init_output_file(ds_name, gkernel, fit_method, dir_output):
	os.makedirs(dir_output, exist_ok=True)
	fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.csv'
	f_detail = open(dir_output + fn_output_detail, 'a')
	csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'edit cost', 
			  'GED method', 'attr distance', 'fit method', 'num graphs', 
			  'target', 'repeat', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
			  'min dis_k gi', 'best kernel dis', 'best graph index', 
			  'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
			  'dis_k gi -> GM', 'edit cost constants', 'time precompute gm',
			  'time optimize ec', 'time generate preimage', 'time total',
			  'itrs', 'converged', 'num updates ecc', 'mge decrease order', 
			  'mge increase order', 'mge converged'])
	f_detail.close()
	
	fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.csv'
	f_summary = open(dir_output + fn_output_summary, 'a')
	csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'edit cost', 
			  'GED method', 'attr distance', 'fit method', 'num graphs', 
			  'target', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
			  'min dis_k gi', 'best kernel dis', 'best graph index',
			  'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
			  'dis_k gi -> GM', 'time precompute gm', 'time optimize ec', 
			  'time generate preimage', 'time total', 'itrs', 'num converged', 
			  'num updates ecc', 'mge num decrease order', 'mge num increase order', 
			  'mge num converged', '# SOD SM -> GM', '# dis_k SM -> GM', 
			  '# dis_k gi -> SM', '# dis_k gi -> GM'])
	f_summary.close()
	
	return fn_output_detail, fn_output_summary


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