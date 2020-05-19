#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:57:18 2020

@author: ljia
"""
import pandas as pd
import numpy as np
import os
import math

def summarize_results_of_random_edit_costs(data_dir, ds_name, gkernel):
	sod_sm_list = []
	sod_gm_list = []
	dis_k_sm_list = []
	dis_k_gm_list = []
	dis_k_min_gi = []
	time_total_list = []
	mge_dec_order_list = []
	mge_inc_order_list = []
	
	# get results from .csv.
	file_name = data_dir + 'results_summary.' + ds_name + '.' + gkernel + '.csv'
	try:
		df = pd.read_csv(file_name)
	except FileNotFoundError:
		return
	for index, row in df.iterrows():
		if row['target'] == 'all' and row['fit method'] == 'random':
			if not math.isnan(float(row['SOD SM'])):
				sod_sm_list.append(float(row['SOD SM']))
			if not math.isnan(float(row['SOD GM'])):
				sod_gm_list.append(float(row['SOD GM']))
			if not math.isnan(float(row['dis_k SM'])):
				dis_k_sm_list.append(float(row['dis_k SM']))
			if not math.isnan(float(row['dis_k GM'])):
				dis_k_gm_list.append(float(row['dis_k GM']))
			if not math.isnan(float(row['min dis_k gi'])):
				dis_k_min_gi.append(float(row['min dis_k gi']))
			if not math.isnan(float(row['time total'])):
				time_total_list.append(float(row['time total']))
			if 'mge num decrease order' in row:
				mge_dec_order_list.append(int(row['mge num decrease order']))
			if 'mge num increase order' in row:
				mge_inc_order_list.append(int(row['mge num increase order']))
	# return if no results.
	if len(sod_sm_list) == 0:
		return
	
	# construct output results.
	op = {}
	op['measure'] = ['max', 'min', 'mean']
	op['SOD SM'] = [np.max(sod_sm_list), np.min(sod_sm_list), np.mean(sod_sm_list)]
	op['SOD GM'] = [np.max(sod_gm_list), np.min(sod_gm_list), np.mean(sod_gm_list)]
	op['dis_k SM'] = [np.max(dis_k_sm_list), np.min(dis_k_sm_list), np.mean(dis_k_sm_list)]
	op['dis_k GM'] = [np.max(dis_k_gm_list), np.min(dis_k_gm_list), np.mean(dis_k_gm_list)]
	op['min dis_k gi'] = [np.max(dis_k_min_gi), np.min(dis_k_min_gi), np.mean(dis_k_min_gi)]
	op['time total'] = [np.max(time_total_list), np.min(time_total_list), np.mean(time_total_list)]
	if len(mge_dec_order_list) > 0:
		op['mge num decrease order'] = [np.max(mge_dec_order_list), np.min(mge_dec_order_list), np.mean(mge_dec_order_list)]
	if len(mge_inc_order_list) > 0:
		op['mge num increase order'] = [np.max(mge_inc_order_list), np.min(mge_inc_order_list), np.mean(mge_inc_order_list)]
	df = pd.DataFrame(data=op)
	
	# write results to .csv
	df.to_csv(data_dir + 'summary_for_random_edit_costs.csv', index=False, header=True)
	
	
def compute_for_all_experiments(data_dir):
	dir_list = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir + i)]
	for dir_name in dir_list:
		sp_tmp = dir_name.split('.')
		ds_name = sp_tmp[0].strip('[error]')
		gkernel = sp_tmp[1]
		summarize_results_of_random_edit_costs(data_dir + dir_name + '/',
										 ds_name, gkernel)
		if os.path.exists(data_dir + dir_name + '/update_order/'):
			summarize_results_of_random_edit_costs(data_dir + dir_name + '/update_order/',
										 ds_name, gkernel)


if __name__ == '__main__':
# 	data_dir = '../results/xp_median_preimage.update_order/'
	data_dir = '../../results/CRIANN/xp_median_preimage.init10/'
	compute_for_all_experiments(data_dir)