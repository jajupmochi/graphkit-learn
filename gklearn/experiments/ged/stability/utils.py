#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:36 2020

@author: ljia
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
import sys
from gklearn.dataset import Dataset
from gklearn.experiments import DATASET_ROOT


def get_dataset(ds_name):
	# The node/edge labels that will not be used in the computation.
#	if ds_name == 'MAO':
#		irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
#	if ds_name == 'Monoterpenoides':
#		irrelevant_labels = {'edge_labels': ['valence']}
#	elif ds_name == 'MUTAG':
#		irrelevant_labels = {'edge_labels': ['label_0']}
	if ds_name == 'AIDS_symb':
		irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']}
		ds_name = 'AIDS'
	else:
		irrelevant_labels = {}

	# Load predefined dataset.
	dataset = Dataset(ds_name, root=DATASET_ROOT)
	# Remove irrelevant labels.
	dataset.remove_labels(**irrelevant_labels)
	print('dataset size:', len(dataset.graphs))
	return dataset


def set_edit_cost_consts(ratio, node_labeled=True, edge_labeled=True, mode='uniform'):
	if mode == 'uniform':
		edit_cost_constants = [i * ratio for i in [1, 1, 1]] + [1, 1, 1]

	if not node_labeled:
		edit_cost_constants[2] = 0
	if not edge_labeled:
		edit_cost_constants[5] = 0

	return edit_cost_constants


def nested_keys_exists(element, *keys):
	'''
	Check if *keys (nested) exists in `element` (dict).
	'''
	if not isinstance(element, dict):
		raise AttributeError('keys_exists() expects dict as first argument.')
	if len(keys) == 0:
		raise AttributeError('keys_exists() expects at least two arguments, one given.')

	_element = element
	for key in keys:
		try:
			_element = _element[key]
		except KeyError:
			return False
	return True


# Check average relative error along elements in two ged matrices.
def matrices_ave_relative_error(m1, m2):
	error = 0
	base = 0
	for i in range(m1.shape[0]):
		for j in range(m1.shape[1]):
			error += np.abs(m1[i, j] - m2[i, j])
# 			base += (np.abs(m1[i, j]) + np.abs(m2[i, j]))
			base += (m1[i, j] + m2[i, j]) # Require only 25% of the time of "base += (np.abs(m1[i, j]) + np.abs(m2[i, j]))".

	base = base / 2

	return error / base


def compute_relative_error(ged_mats):

	if len(ged_mats) != 0:
		# get the smallest "correct" GED matrix.
		ged_mat_s = np.ones(ged_mats[0].shape) * np.inf
		for i in range(ged_mats[0].shape[0]):
			for j in range(ged_mats[0].shape[1]):
				ged_mat_s[i, j] = np.min([mat[i, j] for mat in ged_mats])

		# compute average error.
		errors = []
		for i, mat in enumerate(ged_mats):
			err = matrices_ave_relative_error(mat, ged_mat_s)
		#			 if not per_correct:
		#				 print('matrix # ', str(i))
		#				 pass
			errors.append(err)
	else:
		errors = [0]

	return np.mean(errors)


def parse_group_file_name(fn):
	splits_all = fn.split('.')
	key1 = splits_all[1]

	pos2 = splits_all[2].rfind('_')
#	key2 = splits_all[2][:pos2]
	val2 = splits_all[2][pos2+1:]

	pos3 = splits_all[3].rfind('_')
#	key3 = splits_all[3][:pos3]
	val3 = splits_all[3][pos3+1:] + '.' + splits_all[4]

	return key1, val2, val3


def get_all_errors(save_dir, errors):

	# Loop for each GED matrix file.
	for file in tqdm(sorted(os.listdir(save_dir)), desc='Getting errors', file=sys.stdout):
		if os.path.isfile(os.path.join(save_dir, file)) and file.startswith('ged_mats.'):
			keys = parse_group_file_name(file)

			# Check if the results is in the errors.
			if not keys[0] in errors:
				errors[keys[0]] = {}
			if not keys[1] in errors[keys[0]]:
				errors[keys[0]][keys[1]] = {}
			# Compute the error if not exist.
			if not keys[2] in errors[keys[0]][keys[1]]:
				ged_mats = np.load(os.path.join(save_dir, file))
				errors[keys[0]][keys[1]][keys[2]] = compute_relative_error(ged_mats)

	return errors


def get_relative_errors(save_dir, overwrite=False):
	"""	# Read relative errors from previous computed and saved file. Create the
	file, compute the errors, or add and save the new computed errors to the
	file if necessary.

	Parameters
	----------
	save_dir : TYPE
		DESCRIPTION.
	overwrite : TYPE, optional
		DESCRIPTION. The default is False.

	Returns
	-------
	None.
	"""
	if not overwrite:
		fn_err = save_dir + '/relative_errors.pkl'

		# If error file exists.
		if os.path.isfile(fn_err):
			with open(fn_err, 'rb') as f:
				errors = pickle.load(f)
				errors = get_all_errors(save_dir, errors)
		else:
			errors = get_all_errors(save_dir, {})

	else:
		errors = get_all_errors(save_dir, {})

	with open(fn_err, 'wb') as f:
		pickle.dump(errors, f)

	return errors


def interpolate_result(Z, method='linear'):
	values = Z.copy()
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			if np.isnan(Z[i, j]):

				# Get the nearest non-nan values.
				x_neg = np.nan
				for idx, val in enumerate(Z[i, :][j::-1]):
					if not np.isnan(val):
						x_neg = val
						x_neg_off = idx
						break
				x_pos = np.nan
				for idx, val in enumerate(Z[i, :][j:]):
					if not np.isnan(val):
						x_pos = val
						x_pos_off = idx
						break

				# Interpolate.
				if not np.isnan(x_neg) and not np.isnan(x_pos):
					val_int = (x_pos_off / (x_neg_off + x_pos_off)) * (x_neg - x_pos) + x_pos
					values[i, j] = val_int
					break

				y_neg = np.nan
				for idx, val in enumerate(Z[:, j][i::-1]):
					if not np.isnan(val):
						y_neg = val
						y_neg_off = idx
						break
				y_pos = np.nan
				for idx, val in enumerate(Z[:, j][i:]):
					if not np.isnan(val):
						y_pos = val
						y_pos_off = idx
						break

				# Interpolate.
				if not np.isnan(y_neg) and not np.isnan(y_pos):
					val_int = (y_pos_off / (y_neg_off + y_neg_off)) * (y_neg - y_pos) + y_pos
					values[i, j] = val_int
					break

	return values


def set_axis_style(ax):
	ax.set_axisbelow(True)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.tick_params(labelsize=8, color='w', pad=1, grid_color='w')
	ax.tick_params(axis='x', pad=-2)
	ax.tick_params(axis='y', labelrotation=-40, pad=-2)
#	ax.zaxis._axinfo['juggled'] = (1, 2, 0)
	ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=-3)
	ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=-2, rotation=50)
	ax.set_zlabel(ax.get_zlabel(), fontsize=10, labelpad=-2)
	ax.set_title(ax.get_title(), pad=30, fontsize=15)
	return


def dichotomous_permutation(arr, layer=0):
	import math

# 	def seperate_arr(arr, new_arr):
# 		if (length % 2) == 0:
# 			half = int(length / 2)
# 			new_arr += [arr[half - 1], arr[half]]
# 			subarr1 = [arr[i] for i in range(1, half - 1)]
# 		else:
# 			half = math.floor(length / 2)
# 			new_arr.append(arr[half])
# 			subarr1 = [arr[i] for i in range(1, half)]
# 		subarr2 = [arr[i] for i in range(half + 1, length - 1)]
# 		subarrs = [subarr1, subarr2]
# 		return subarrs


	if layer == 0:
		length = len(arr)
		if length <= 2:
			return arr

		new_arr = [arr[0], arr[-1]]
		if (length % 2) == 0:
 			half = int(length / 2)
 			new_arr += [arr[half - 1], arr[half]]
 			subarr1 = [arr[i] for i in range(1, half - 1)]
		else:
 			half = math.floor(length / 2)
 			new_arr.append(arr[half])
 			subarr1 = [arr[i] for i in range(1, half)]
		subarr2 = [arr[i] for i in range(half + 1, length - 1)]
		subarrs = [subarr1, subarr2]
# 		subarrs = seperate_arr(arr, new_arr)
		new_arr += dichotomous_permutation(subarrs, layer=layer+1)

	else:
		new_arr = []
		subarrs = []
		for a in arr:
			length = len(a)
			if length <= 2:
				new_arr += a
			else:
# 				subarrs += seperate_arr(a, new_arr)
				if (length % 2) == 0:
 					half = int(length / 2)
 					new_arr += [a[half - 1], a[half]]
 					subarr1 = [a[i] for i in range(0, half - 1)]
				else:
 					half = math.floor(length / 2)
 					new_arr.append(a[half])
 					subarr1 = [a[i] for i in range(0, half)]
				subarr2 = [a[i] for i in range(half + 1, length)]
				subarrs += [subarr1, subarr2]

		if len(subarrs) > 0:
			new_arr += dichotomous_permutation(subarrs, layer=layer+1)

	return new_arr

# 	length = len(arr)
# 	if length <= 2:
# 		return arr

# 	new_arr = [arr[0], arr[-1]]
# 	if (length % 2) == 0:
# 		half = int(length / 2)
# 		new_arr += [arr[half - 1], arr[half]]
# 		subarr1 = [arr[i] for i in range(1, half - 1)]
# 	else:
# 		half = math.floor(length / 2)
# 		new_arr.append(arr[half])
# 		subarr1 = [arr[i] for i in range(1, half)]
# 	subarr2 = [arr[i] for i in range(half + 1, length - 1)]
# 	if len(subarr1) > 0:
# 		new_arr += dichotomous_permutation(subarr1)
# 	if len(subarr2) > 0:
# 		new_arr += dichotomous_permutation(subarr2)

# 	return new_arr


def mix_param_grids(list_of_grids):
	mixed_grids = []
	not_finished = [True] * len(list_of_grids)
	idx = 0
	while sum(not_finished) > 0:
		for g_idx, grid in enumerate(list_of_grids):
			if idx < len(grid):
				mixed_grids.append(grid[idx])
			else:
				not_finished[g_idx] = False
		idx += 1

	return mixed_grids



if __name__ == '__main__':
	root_dir = 'outputs/CRIANN/'
#	for dir_ in sorted(os.listdir(root_dir)):
#		if os.path.isdir(root_dir):
#			full_dir = os.path.join(root_dir, dir_)
#			print('---', full_dir,':')
#			save_dir = os.path.join(full_dir, 'groups/')
#			if os.path.exists(save_dir):
#				try:
#					get_relative_errors(save_dir)
#				except Exception as exp:
#					print('An exception occured when running this experiment:')
#					print(repr(exp))
