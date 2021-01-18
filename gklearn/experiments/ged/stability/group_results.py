#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:26:43 2020

@author: ljia

This script groups results together into a single file for the sake of faster
searching and loading.
"""
import os
import pickle
import numpy as np
from shutil import copyfile
from tqdm import tqdm
import sys


def check_group_existence(file_name):
	path, name = os.path.split(file_name)
	marker_fn = os.path.join(path, 'group_names_finished.pkl')
	if os.path.isfile(marker_fn):
		with open(marker_fn, 'rb') as f:
			fns = pickle.load(f)
			if name in fns:
				return True

	if os.path.isfile(file_name):
		return True

	return False


def update_group_marker(file_name):
	path, name = os.path.split(file_name)
	marker_fn = os.path.join(path, 'group_names_finished.pkl')
	if os.path.isfile(marker_fn):
		with open(marker_fn, 'rb') as f:
			fns = pickle.load(f)
			if name in fns:
				return
			else:
				fns.add(name)
	else:
		fns = set({name})
	with open(marker_fn, 'wb') as f:
		pickle.dump(fns, f)


def create_group_marker_file(dir_folder, overwrite=True):
	if not overwrite:
		return

	fns = set()
	for file in sorted(os.listdir(dir_folder)):
		if os.path.isfile(os.path.join(dir_folder, file)):
			if file.endswith('.npy'):
				fns.add(file)

	marker_fn = os.path.join(dir_folder, 'group_names_finished.pkl')
	with open(marker_fn, 'wb') as f:
		pickle.dump(fns, f)


# This function is used by other scripts. Modify it carefully.
def group_trials(dir_folder, name_prefix, overwrite, clear, backup, num_trials=100):

	# Get group name.
	label_name = name_prefix.split('.')[0]
	if label_name == 'ged_matrix':
		group_label = 'ged_mats'
	elif label_name == 'runtime':
		group_label = 'runtimes'
	else:
		group_label = label_name
	name_suffix = name_prefix[len(label_name):]
	if label_name == 'ged_matrix':
		name_group = dir_folder + 'groups/' + group_label + name_suffix + 'npy'
	else:
		name_group = dir_folder + 'groups/' + group_label + name_suffix + 'pkl'

	if not overwrite and os.path.isfile(name_group):
		# Check if all trial files exist.
		trials_complete = True
		for trial in range(1, num_trials + 1):
			file_name = dir_folder + name_prefix + 'trial_' + str(trial) + '.pkl'
			if not os.path.isfile(file_name):
				trials_complete = False
				break
	else:
		# Get data.
		data_group = []
		for trial in range(1, num_trials + 1):
			file_name = dir_folder + name_prefix + 'trial_' + str(trial) + '.pkl'
			if os.path.isfile(file_name):
				with open(file_name, 'rb') as f:
					try:
						data = pickle.load(f)
					except EOFError:
						print('EOF Error occurred.')
						return
					data_group.append(data)

# 					unpickler = pickle.Unpickler(f)
# 					data = unpickler.load()
# 					if not isinstance(data, np.array):
# 						return
# 					else:
# 						data_group.append(data)

			else: # Not all trials are completed.
				return

		# Write groups.
		if label_name == 'ged_matrix':
			data_group = np.array(data_group)
			with open(name_group, 'wb') as f:
				np.save(f, data_group)
		else:
			with open(name_group, 'wb') as f:
				pickle.dump(data_group, f)

		trials_complete = True

	if trials_complete:
		# Backup.
		if backup:
			for trial in range(1, num_trials + 1):
				src = dir_folder + name_prefix + 'trial_' + str(trial) + '.pkl'
				dst = dir_folder + 'backups/' + name_prefix + 'trial_' + str(trial) + '.pkl'
				copyfile(src, dst)

		# Clear.
		if clear:
			for trial in range(1, num_trials + 1):
				src = dir_folder + name_prefix + 'trial_' + str(trial) + '.pkl'
				os.remove(src)


def group_all_in_folder(dir_folder, overwrite=False, clear=True, backup=True):

	# Create folders.
	os.makedirs(dir_folder + 'groups/', exist_ok=True)
	if backup:
		os.makedirs(dir_folder + 'backups', exist_ok=True)

	# Iterate all files.
	cur_file_prefix = ''
	for file in tqdm(sorted(os.listdir(dir_folder)), desc='Grouping', file=sys.stdout):
		if os.path.isfile(os.path.join(dir_folder, file)):
			name_prefix = file.split('trial_')[0]
# 			print(name)
# 			print(name_prefix)
			if name_prefix != cur_file_prefix:
				group_trials(dir_folder, name_prefix, overwrite, clear, backup)
				cur_file_prefix = name_prefix



if __name__ == '__main__':
 	# dir_folder = 'outputs/CRIANN/edit_costs.num_sols.ratios.IPFP/'
 	# group_all_in_folder(dir_folder)

 	# dir_folder = 'outputs/CRIANN/edit_costs.repeats.ratios.IPFP/'
 	# group_all_in_folder(dir_folder)

 	# dir_folder = 'outputs/CRIANN/edit_costs.max_num_sols.ratios.bipartite/'
 	# group_all_in_folder(dir_folder)

 	# dir_folder = 'outputs/CRIANN/edit_costs.repeats.ratios.bipartite/'
 	# group_all_in_folder(dir_folder)

	dir_folder = 'outputs/CRIANN/edit_costs.real_data.num_sols.ratios.IPFP/groups/'
	create_group_marker_file(dir_folder)