#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:16:33 2020

@author: ljia
"""
import pandas as pd
import numpy as np
import os


DS_SYMB = ['MUTAG', 'Monoterpenoides', 'MAO_symb']
DS_NON_SYMB = ['Letter-high', 'Letter-med', 'Letter-low', 'COIL-RAG', 'PAH']
DS_UNLABELED = ['PAH_unlabeled']


def rounder(x, decimals):
	x_strs = str(x).split('.')
	if len(x_strs) == 2:
		before = x_strs[0]
		after = x_strs[1]
		if len(after) > decimals:
			if int(after[decimals]) >= 5:
				after0s = ''
				for c in after:
					if c == '0':
						after0s += '0'
					elif c != '0':
						break
				after = after0s + str(int(after[0:decimals]) + 1)[-decimals:]
			else:
				after = after[0:decimals]
		elif len(after) < decimals:
			after += '0' * (decimals - len(after))
		return before + '.' + after

	elif len(x_strs) == 1:
		return x_strs[0]
	

def replace_nth(string, sub, wanted, n):
	import re
	where = [m.start() for m in re.finditer(sub, string)][n-1]
	before = string[:where]
	after = string[where:]
	after = after.replace(sub, wanted, 1)
	newString = before + after
	return newString


def df_to_latex_table(df):
	ltx = df.to_latex(index=True, escape=False, multirow=True)
	
	# modify middle lines.
	ltx = ltx.replace('\\cline{1-9}\n\\cline{2-9}', 	'\\toprule')
	ltx = ltx.replace('\\cline{2-9}', '\\cmidrule(l){2-9}')
	
	# modify first row.
	i_start = ltx.find('\n\\toprule\n')
	i_end = ltx.find('\\\\\n\\midrule\n')
	ltx = ltx.replace(ltx[i_start:i_end+12], '\n\\toprule\nDatasets & Graph Kernels & Algorithms & $d_\\mathcal{F}$ SM & $d_\\mathcal{F}$ SM (UO) & $d_\\mathcal{F}$ GM & $d_\\mathcal{F}$ GM (UO) & Runtime & Runtime (UO) \\\\\n\\midrule\n', 1)
	
	# add row numbers.
	ltx = ltx.replace('lllllllll', 'lllllllll|@{\\makebox[2em][r]{\\textit{\\rownumber\\space}}}', 1)
	ltx = replace_nth(ltx, '\\\\\n', '\\gdef\\rownumber{\\stepcounter{magicrownumbers}\\arabic{magicrownumbers}} \\\\\n', 1)
		
	return ltx


def beautify_df(df):
	df = df.sort_values(by=['Datasets', 'Graph Kernels'])
	df = df.set_index(['Datasets', 'Graph Kernels', 'Algorithms'])
# 	index = pd.MultiIndex.from_frame(df[['Datasets', 'Graph Kernels', 'Algorithms']])

	# bold the best results.
	for ds in df.index.get_level_values('Datasets').unique():
		for gk in df.loc[ds].index.get_level_values('Graph Kernels').unique():
			min_val = np.inf
			min_indices = []
			min_labels = []
			for index, row in df.loc[(ds, gk)].iterrows():
				for label in ['$d_\mathcal{F}$ SM', '$d_\mathcal{F}$ GM', '$d_\mathcal{F}$ GM (UO)']:
					value = row[label]
					if value != '-':
						value = float(value.strip('/same'))
						if value < min_val:
							min_val = value
							min_indices = [index]
							min_labels = [label]
						elif value == min_val:
							min_indices.append(index)
							min_labels.append(label)
			for idx, index in enumerate(min_indices):
				df.loc[(ds, gk, index), min_labels[idx]] = '\\textbf{' + df.loc[(ds, gk, index), min_labels[idx]] + '}'
	
	return df


def get_results(data_dir, ds_name, gkernel):
	# get results from .csv.
	file_name = data_dir + 'results_summary.' + ds_name + '.' + gkernel + '.csv'
	try:
		df_summary = pd.read_csv(file_name)
	except FileNotFoundError:
		return None

	df_results = pd.DataFrame(index=None, columns=['d_F SM', 'd_F GM', 'runtime'])
	for index, row in df_summary.iterrows():
		if row['target'] == 'all' and row['fit method'] == 'k-graphs':
			df_results.loc['From median set'] = ['-', rounder(row['min dis_k gi'], 3), '-']
			if_uo = (int(row['mge num decrease order']) > 0 or int(row['mge num increase order']) > 0)
			df_results.loc['Optimized'] = [rounder(row['dis_k SM'], 3), 
								  rounder(row['dis_k GM'], 3) if if_uo else (rounder(row['dis_k GM'], 3) + '/same'),
								  rounder(row['time total'], 2)]
		if row['target'] == 'all' and row['fit method'] == 'expert':
			if_uo = (int(row['mge num decrease order']) > 0 or int(row['mge num increase order']) > 0)
			df_results.loc['IAM: expert costs'] = [rounder(row['dis_k SM'], 3), 
								  rounder(row['dis_k GM'], 3) if if_uo else (rounder(row['dis_k GM'], 3) + '/same'),
								  rounder(row['time total'], 2)]
								  
	# get results from random summary .csv.
	random_fini = True
	file_name = data_dir + 'summary_for_random_edit_costs.csv'
	try:
		df_random = pd.read_csv(file_name)
	except FileNotFoundError:
		random_fini = False

	if random_fini:								  
		for index, row in df_random.iterrows():
			if row['measure'] == 'mean':
				if_uo = (float(row['mge num decrease order']) > 0 or float(row['mge num increase order']) > 0)
				df_results.loc['IAM: random costs'] = [rounder(row['dis_k SM'], 3), 
								  rounder(row['dis_k GM'], 3) if if_uo else (rounder(row['dis_k GM'], 3) + '/same'),
								  rounder(row['time total'], 2)]
				
	# sort index.
	df_results = df_results.reindex([item for item in ['From median set', 'IAM: random costs', 'IAM: expert costs', 'Optimized'] if item in df_results.index])
				
	return df_results
	

def get_results_of_one_xp(data_dir, ds_name, gkernel):
	df_results = pd.DataFrame()
	
	df_tmp_uo = None
	if not os.path.isfile(data_dir + 'update_order/error.txt'):
		df_tmp_uo = get_results(data_dir + 'update_order/', ds_name, gkernel)

	df_tmp = None
	if not os.path.isfile(data_dir + 'error.txt'):
		df_tmp = get_results(data_dir, ds_name, gkernel)

	if (df_tmp_uo is not None and not df_tmp_uo.empty) or (df_tmp is not None and not df_tmp.empty):
		df_results = pd.DataFrame(index=['From median set', 'IAM: random costs', 'IAM: expert costs', 'Optimized'], columns=['$d_\mathcal{F}$ SM', '$d_\mathcal{F}$ SM (UO)', '$d_\mathcal{F}$ GM', '$d_\mathcal{F}$ GM (UO)', 'Runtime', 'Runtime (UO)'])
		if df_tmp_uo is not None and not df_tmp_uo.empty:
			for index, row in df_tmp_uo.iterrows():
				for algo in df_results.index:
					if index == algo:
						df_results.at[algo, '$d_\mathcal{F}$ SM (UO)'] = row['d_F SM']
						df_results.at[algo, '$d_\mathcal{F}$ GM (UO)'] = row['d_F GM']
						df_results.at[algo, 'Runtime (UO)'] = row['runtime']
		if df_tmp is not None and not df_tmp.empty:
			for index, row in df_tmp.iterrows():
				for algo in df_results.index:
					if index == algo:
						df_results.at[algo, '$d_\mathcal{F}$ SM'] = row['d_F SM']
						df_results.at[algo, '$d_\mathcal{F}$ GM'] = row['d_F GM'].strip('/same')
						df_results.at[algo, 'Runtime'] = row['runtime']	
	
	df_results = df_results.dropna(axis=0, how='all')
	df_results = df_results.fillna(value='-')
	df_results = df_results.reset_index().rename(columns={'index': 'Algorithms'})
		
	return df_results


def get_results_for_all_experiments(root_dir):
	columns=['Datasets', 'Graph Kernels', 'Algorithms', '$d_\mathcal{F}$ SM', '$d_\mathcal{F}$ SM (UO)', '$d_\mathcal{F}$ GM', '$d_\mathcal{F}$ GM (UO)', 'Runtime', 'Runtime (UO)']
	df_symb = pd.DataFrame(columns=columns)
	df_nonsymb = pd.DataFrame(columns=columns)
	df_unlabeled = pd.DataFrame(columns=columns)
	
	dir_list = [i for i in os.listdir(root_dir) if os.path.isdir(root_dir + i)]
	for dir_name in dir_list:
		sp_tmp = dir_name.split('.')
		gkernel = sp_tmp[1]
		ds_name = sp_tmp[0].strip('[error]')
		suffix = ''
		if sp_tmp[-1] == 'unlabeled':
 			suffix = '_unlabeled'
		elif sp_tmp[-1] == 'symb':
 			suffix = '_symb'
			 
		df_results = get_results_of_one_xp(root_dir + dir_name + '/', ds_name, gkernel)
		
		if not df_results.empty:
			ds_name += suffix
			if ds_name in DS_SYMB:
				for index, row in df_results.iterrows():
					df_symb.loc[len(df_symb)] = [ds_name.replace('_', '\_'), gkernel] + row.tolist()
			elif ds_name in DS_NON_SYMB:
				for index, row in df_results.iterrows():
					df_nonsymb.loc[len(df_nonsymb)] = [ds_name.replace('_', '\_'), gkernel] + row.tolist()
			elif ds_name in DS_UNLABELED:
				for index, row in df_results.iterrows():
					df_unlabeled.loc[len(df_unlabeled)] = [ds_name.replace('_', '\_'), gkernel] + row.tolist()
			else:
				raise Exception('dataset' + ds_name + 'is not pre-defined.')
				
	# sort.
	df_symb = beautify_df(df_symb)
	df_nonsymb = beautify_df(df_nonsymb)
	df_unlabeled = beautify_df(df_unlabeled)
	
	# convert dfs to latex strings.
	ltx_symb = df_to_latex_table(df_symb)
	ltx_nonsymb = df_to_latex_table(df_nonsymb)
	ltx_unlabeled = df_to_latex_table(df_unlabeled)
	
	return ltx_symb, ltx_nonsymb, ltx_unlabeled


if __name__ == '__main__':
# 	root_dir = '../results/xp_median_preimage.init20/'
	root_dir = '../../results/CRIANN/xp_median_preimage.init10/'
	ltx_symb, ltx_nonsymb, ltx_unlabeled = get_results_for_all_experiments(root_dir)
