#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:02:51 2020

@author: ljia
"""

import numpy as np
from gklearn.utils import Dataset
import csv
import os
import os.path
from gklearn.preimage import RandomPreimageGenerator
from gklearn.utils import split_dataset_by_target
from gklearn.utils.graphfiles import saveGXL


def generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=True, save_preimages=True, load_gm='auto', dir_save='', irrelevant_labels=None, edge_required=False, cut_range=None):	
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
		fn_output_detail, fn_output_summary = _init_output_file_preimage(ds_name, kernel_options['name'], dir_save)
		

	dis_k_dataset_list = []
	dis_k_preimage_list = []
	time_precompute_gm_list = []
	time_generate_list = []
	time_total_list = []
	itrs_list = []
	num_updates_list = []
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
				rpg_options['gram_matrix_unnorm'] = gram_matrix_unnorm_list[idx - idx_offset]
				rpg_options['runtime_precompute_gm'] = time_precompute_gm_list[idx - idx_offset]
		rpg = RandomPreimageGenerator()
		rpg.dataset = dataset
		rpg.set_options(**rpg_options.copy())
		rpg.kernel_options = kernel_options.copy()

		# 3. compute preimage.
		print('3. computing preimage...')
		rpg.run()
		results = rpg.get_results()
		
		# 4. save results (and median graphs).
		print('4. saving results (and preimages)...')
		# write result detail.
		if save_results:
			print('writing results to files...')

			f_detail = open(dir_save + fn_output_detail, 'a')
			csv.writer(f_detail).writerow([ds_name, kernel_options['name'], 
					  num_graphs, target, 1, 
					  results['k_dis_dataset'], results['k_dis_preimage'],
					  results['runtime_precompute_gm'], 
					  results['runtime_generate_preimage'], results['runtime_total'],
					  results['itrs'], results['num_updates']])
			f_detail.close()
		
			# compute result summary.
			dis_k_dataset_list.append(results['k_dis_dataset'])
			dis_k_preimage_list.append(results['k_dis_preimage'])
			time_precompute_gm_list.append(results['runtime_precompute_gm'])
			time_generate_list.append(results['runtime_generate_preimage'])
			time_total_list.append(results['runtime_total'])
			itrs_list.append(results['itrs'])
			num_updates_list.append(results['num_updates'])
			
			# write result summary for each letter.
			f_summary = open(dir_save + fn_output_summary, 'a')
			csv.writer(f_summary).writerow([ds_name, kernel_options['name'], 
					  num_graphs, target,
					  results['k_dis_dataset'], results['k_dis_preimage'],
					  results['runtime_precompute_gm'], 
					  results['runtime_generate_preimage'], results['runtime_total'],
					  results['itrs'], results['num_updates']])
			f_summary.close()
			 
		# save median graphs.
		if save_preimages:
			os.makedirs(dir_save + 'preimages/', exist_ok=True)
			print('Saving preimages to files...')
			fn_best_dataset = dir_save + 'preimages/g_best_dataset.' + 'nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(rpg.best_from_dataset, fn_best_dataset + '.gxl', method='default', 
					node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
			fn_preimage = dir_save + 'preimages/g_preimage.' + 'nbg' + str(num_graphs) + '.y' + str(target) + '.repeat' + str(1)
			saveGXL(rpg.preimage, fn_preimage + '.gxl', method='default',
		  			node_labels=dataset.node_labels, edge_labels=dataset.edge_labels,	
					node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs)
				
		if (load_gm == 'auto' and not gmfile_exist) or not load_gm:
			gram_matrix_unnorm_list.append(rpg.gram_matrix_unnorm)

	# write result summary for each class. 
	if save_results:
		dis_k_dataset_mean = np.mean(dis_k_dataset_list)
		dis_k_preimage_mean = np.mean(dis_k_preimage_list)
		time_precompute_gm_mean = np.mean(time_precompute_gm_list)
		time_generate_mean = np.mean(time_generate_list)
		time_total_mean = np.mean(time_total_list)
		itrs_mean = np.mean(itrs_list)
		num_updates_mean = np.mean(num_updates_list)
		f_summary = open(dir_save + fn_output_summary, 'a')
		csv.writer(f_summary).writerow([ds_name, kernel_options['name'],
				  num_graphs, 'all',
				  dis_k_dataset_mean, dis_k_preimage_mean,
				  time_precompute_gm_mean,
				  time_generate_mean, time_total_mean, itrs_mean, 
				  num_updates_mean])
		f_summary.close()
	
	# write Gram matrices to file.
	if (load_gm == 'auto' and not gmfile_exist) or not load_gm:
		np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm_list=gram_matrix_unnorm_list, run_time_list=time_precompute_gm_list)	

	print('\ncomplete.\n')	

	
def _init_output_file_preimage(ds_name, gkernel, dir_output):
	os.makedirs(dir_output, exist_ok=True)
	fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.csv'
	f_detail = open(dir_output + fn_output_detail, 'a')
	csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'num graphs', 
			  'target', 'repeat', 'dis_k best from dataset', 'dis_k preimage',
			  'time precompute gm', 'time generate preimage', 'time total',
			  'itrs', 'num updates'])
	f_detail.close()
	
	fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.csv'
	f_summary = open(dir_output + fn_output_summary, 'a')
	csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'num graphs', 
			  'target', 'dis_k best from dataset', 'dis_k preimage',
			  'time precompute gm', 'time generate preimage', 'time total',
			  'itrs', 'num updates'])
	f_summary.close()
	
	return fn_output_detail, fn_output_summary