#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:49:43 2020

@author: ljia
"""

import os
import re
import pickle


OUT_TIME_LIST = []


OUT_MEM_LIST = set({('ShortestPath', 'REDDIT-BINARY', 'True'),
				('ShortestPath', 'REDDIT-BINARY', 'False'),
				('StructuralSP', 'ENZYMES', 'False'),
				('StructuralSP', 'AIDS', 'False'),
				('ShortestPath', 'DD', 'True'),
				('ShortestPath', 'DD', 'False'),
				('StructuralSP', 'DD', 'True'),
				('StructuralSP', 'DD', 'False'),
				('StructuralSP', 'COIL-DEL', 'True'),
				('ShortestPath', 'COLORS-3', 'True'),
				('ShortestPath', 'COLORS-3', 'False'),
				('StructuralSP', 'COLORS-3', 'True'),
				('StructuralSP', 'COLORS-3', 'False'),
				('StructuralSP', 'PROTEINS', 'True'),
				('StructuralSP', 'PROTEINS', 'False'),
				('StructuralSP', 'PROTEINS_full', 'True'),
				('StructuralSP', 'PROTEINS_full', 'False'),
				('StructuralSP', 'MSRC_21', 'False'),
				('ShortestPath', 'MCF-7', 'True'),
				('ShortestPath', 'MCF-7', 'False'),
				('StructuralSP', 'MCF-7', 'True'),
				('StructuralSP', 'MCF-7', 'False'),
				('ShortestPath', 'MCF-7H', 'True'),
				('ShortestPath', 'MCF-7H', 'False'),
				('StructuralSP', 'MCF-7H', 'True'),
				('StructuralSP', 'MCF-7H', 'False'),
				('ShortestPath', 'MOLT-4', 'True'),
				('ShortestPath', 'MOLT-4', 'False'),
				('StructuralSP', 'MOLT-4', 'True'),
				('StructuralSP', 'MOLT-4', 'False'),
				('ShortestPath', 'MOLT-4H', 'True'),
				('ShortestPath', 'MOLT-4H', 'False'),
				('StructuralSP', 'MOLT-4H', 'True'),
				('StructuralSP', 'MOLT-4H', 'False'),
				('ShortestPath', 'P388', 'True'),
				('ShortestPath', 'P388', 'False'),
				('StructuralSP', 'P388', 'True'),
				('StructuralSP', 'P388', 'False'),
				('ShortestPath', 'P388H', 'True'),
				('ShortestPath', 'P388H', 'False'),
				('StructuralSP', 'P388H', 'True'),
				('StructuralSP', 'P388H', 'False'),
				('StructuralSP', 'NCI1', 'False'),
				('ShortestPath', 'NCI-H23', 'True'),
				('ShortestPath', 'NCI-H23', 'False'),
				('StructuralSP', 'NCI-H23', 'True'),
				('StructuralSP', 'NCI-H23', 'False'),
				('ShortestPath', 'NCI-H23H', 'True'),
				('ShortestPath', 'NCI-H23H', 'False'),
				('StructuralSP', 'NCI-H23H', 'True'),
				('StructuralSP', 'NCI-H23H', 'False'),
				('StructuralSP', 'OHSU', 'False'),
				('ShortestPath', 'OVCAR-8', 'True'),
				('ShortestPath', 'OVCAR-8', 'False'),
				('StructuralSP', 'OVCAR-8', 'True'),
				('StructuralSP', 'OVCAR-8', 'False'),
				('ShortestPath', 'OVCAR-8H', 'True'),
				('ShortestPath', 'OVCAR-8H', 'False'),
				('StructuralSP', 'OVCAR-8H', 'True'),
				('StructuralSP', 'OVCAR-8H', 'False'),
				('ShortestPath', 'SN12C', 'True'),
				('ShortestPath', 'SN12C', 'False'),
				('StructuralSP', 'SN12C', 'True'),
				('StructuralSP', 'SN12C', 'False'),
				('ShortestPath', 'SN12CH', 'True'),
				('ShortestPath', 'SN12CH', 'False'),
				('ShortestPath', 'SF-295', 'True'),
				('ShortestPath', 'SF-295', 'False'),
				('StructuralSP', 'SF-295', 'True'),
				('StructuralSP', 'SF-295', 'False'),
				('ShortestPath', 'SF-295H', 'True'),
				('ShortestPath', 'SF-295H', 'False'),
				('StructuralSP', 'SF-295H', 'True'),
				('StructuralSP', 'SF-295H', 'False'),
				('ShortestPath', 'SW-620', 'True'),
				('ShortestPath', 'SW-620', 'False'),
				('StructuralSP', 'SW-620', 'True'),
				('StructuralSP', 'SW-620', 'False'),
				('ShortestPath', 'SW-620H', 'True'),
				('ShortestPath', 'SW-620H', 'False'),
				('StructuralSP', 'SW-620H', 'True'),
				('StructuralSP', 'SW-620H', 'False'),
				('ShortestPath', 'TRIANGLES', 'True'),
				('ShortestPath', 'TRIANGLES', 'False'),
				('StructuralSP', 'TRIANGLES', 'True'),
				('StructuralSP', 'TRIANGLES', 'False'),
				('ShortestPath', 'Yeast', 'True'),
				('ShortestPath', 'Yeast', 'False'),
				('StructuralSP', 'Yeast', 'True'),
				('StructuralSP', 'Yeast', 'False'),
				('ShortestPath', 'YeastH', 'True'),
				('ShortestPath', 'YeastH', 'False'),
				('StructuralSP', 'YeastH', 'True'),
				('StructuralSP', 'YeastH', 'False'),
				('ShortestPath', 'FRANKENSTEIN', 'True'),
				('ShortestPath', 'FRANKENSTEIN', 'False'),
				('StructuralSP', 'FRANKENSTEIN', 'True'),
				('StructuralSP', 'FRANKENSTEIN', 'False'),
				('StructuralSP', 'SN12CH', 'True'),
				('StructuralSP', 'SN12CH', 'False'),
				('ShortestPath', 'UACC257', 'True'),
				('ShortestPath', 'UACC257', 'False'),
				('StructuralSP', 'UACC257', 'True'),
				('StructuralSP', 'UACC257', 'False'),
				('ShortestPath', 'UACC257H', 'True'),
				('ShortestPath', 'UACC257H', 'False'),
				('StructuralSP', 'UACC257H', 'True'),
				('StructuralSP', 'UACC257H', 'False'),
				('ShortestPath', 'PC-3', 'True'),
				('ShortestPath', 'PC-3', 'False'),
				('StructuralSP', 'PC-3', 'True'),
				('StructuralSP', 'PC-3', 'False'),
				('ShortestPath', 'PC-3H', 'True'),
				('ShortestPath', 'PC-3H', 'False'),
				('StructuralSP', 'PC-3H', 'True'),
				('StructuralSP', 'PC-3H', 'False'),
				('ShortestPath', 'DBLP_v1', 'True'),
				('ShortestPath', 'DBLP_v1', 'False'),
				('StructuralSP', 'DBLP_v1', 'True'),
				('StructuralSP', 'DBLP_v1', 'False'),
				('ShortestPath', 'COLLAB', 'True'),
				('ShortestPath', 'COLLAB', 'False'),
				('StructuralSP', 'COLLAB', 'True'),
				('StructuralSP', 'COLLAB', 'False'),
				('ShortestPath', 'REDDIT-BINARY', 'False'),
				('StructuralSP', 'REDDIT-BINARY', 'True'),
				('StructuralSP', 'REDDIT-BINARY', 'False'),
				('ShortestPath', 'REDDIT-MULTI-5K', 'True'),
				('ShortestPath', 'REDDIT-MULTI-5K', 'False'),
				('StructuralSP', 'REDDIT-MULTI-5K', 'True'),
				('StructuralSP', 'REDDIT-MULTI-5K', 'False'),
				('ShortestPath', 'REDDIT-MULTI-12K', 'True'),
				('ShortestPath', 'REDDIT-MULTI-12K', 'False'),
				('StructuralSP', 'REDDIT-MULTI-12K', 'True'),
				('StructuralSP', 'REDDIT-MULTI-12K', 'False'),
				('ShortestPath', 'TWITTER-Real-Graph-Partial', 'True'),
				('ShortestPath', 'TWITTER-Real-Graph-Partial', 'False'),
				('StructuralSP', 'TWITTER-Real-Graph-Partial', 'True'),
				('StructuralSP', 'TWITTER-Real-Graph-Partial', 'False'),
				})

MISS_LABEL_LIST = set({('StructuralSP', 'GREC', 'True'),
				   ('StructuralSP', 'GREC', 'False'),
				   ('StructuralSP', 'Web', 'True'),
				   ('StructuralSP', 'Web', 'False'),
				   })


def get_job_script(kernel, dataset, fcsp):
# 	if (kernel, dataset, fcsp) in OUT_MEM_LIST:
# 		mem = '2560000'
# 	else:
	mem = '4000'
	script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name="fcsp.space.""" + kernel + r"." + dataset + r"." + fcsp + r""""
#SBATCH --partition=""" + (r"court" if kernel == 'ShortestPath' else r"court") + r"""
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_fcsp.space.""" + kernel + r"." + dataset + r"." + fcsp + r""".txt"
#SBATCH --error="errors/error_fcsp.space.""" + kernel + r"." + dataset + r"." + fcsp + r""".txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=""" + (r"48" if kernel == 'ShortestPath' else r"48") + r""":00:00
##SBATCH --mem-per-cpu=""" + mem + r"""
#SBATCH --mem=4000

srun hostname
srun cd /home/2019015/ljia02/graphkit-learn/gklearn/experiments/thesis/graph_kernels/fcsp
srun python3 compare_fcsp_space.py """ + kernel + r" " + dataset + r" " + fcsp
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return script


def check_task_status(save_dir, *params):
	str_task_id = '.' + '.'.join(params)

	# Check if the task is in out of memeory or out of space lists or missing labels.
	if params in OUT_MEM_LIST or params in OUT_TIME_LIST or params in MISS_LABEL_LIST:
		return True

	# Check if the task is running or in queue of slurm.
	command = 'squeue --user $USER --name "fcsp.space' + str_task_id + '" --format "%.2t" --noheader'
	stream = os.popen(command)
	output = stream.readlines()
	if len(output) > 0:
		return True

	# Check if the task is already computed.
	file_name = os.path.join(save_dir, 'space' + str_task_id + '.pkl')
	if os.path.getsize(file_name) > 0:
		if os.path.isfile(file_name):
			with open(file_name, 'rb') as f:
				data = pickle.load(f)
				if data['completed']:
					return True

	return False


if __name__ == '__main__':
	save_dir = 'outputs/'
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	from sklearn.model_selection import ParameterGrid

	Dataset_List = ['Alkane_unlabeled', 'Alkane', 'Acyclic', 'MAO_lite', 'MAO',
				    'PAH_unlabeled', 'PAH', 'MUTAG', 'Monoterpens',
					'Letter-high', 'Letter-med', 'Letter-low',
					'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD',
					# new: not so large.
					'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR', 'Chiral', 'Vitamin_D',
					'ACE', 'Steroid', 'KKI', 'Fingerprint', 'IMDB-BINARY',
					'IMDB-MULTI', 'Peking_1', 'Cuneiform', 'OHSU', 'BZR', 'COX2',
					'DHFR', 'SYNTHETICnew', 'Synthie', 'SYNTHETIC',
					# new: large.
					'TWITTER-Real-Graph-Partial', 'GREC', 'Web', 'MCF-7',
					'MCF-7H', 'MOLT-4', 'MOLT-4H', 'NCI-H23', 'NCI-H23H',
					'OVCAR-8', 'OVCAR-8H', 'P388', 'P388H', 'PC-3', 'PC-3H',
					'SF-295', 'SF-295H', 'SN12C', 'SN12CH', 'SW-620', 'SW-620H',
					'TRIANGLES', 'UACC257', 'UACC257H', 'Yeast', 'YeastH',
					'COLORS-3', 'DBLP_v1', 'REDDIT-MULTI-12K',
					'REDDIT-MULTI-12K', 'REDDIT-MULTI-12K',
					'REDDIT-MULTI-12K', 'MSRC_9', 'MSRC_21', 'MSRC_21C',
					'COLLAB', 'COIL-DEL',
					'COIL-RAG', 'PROTEINS', 'PROTEINS_full', 'Mutagenicity',
					'REDDIT-BINARY', 'FRANKENSTEIN', 'REDDIT-MULTI-5K',
					'REDDIT-MULTI-12K']

	Kernel_List = ['ShortestPath', 'StructuralSP']

	fcsp_list = ['True', 'False']

	task_grid = ParameterGrid({'kernel': Kernel_List[:],
							'dataset': Dataset_List[:],
							'fcsp': fcsp_list[:]})

	from tqdm import tqdm

	for task in tqdm(list(task_grid), desc='submitting tasks/jobs'):

		if False == check_task_status(save_dir, task['kernel'], task['dataset'], task['fcsp']):
			job_script = get_job_script(task['kernel'], task['dataset'], task['fcsp'])
			command = 'sbatch <<EOF\n' + job_script + '\nEOF'
	# 			print(command)
			os.system(command)
	# 		os.popen(command)
	# 		output = stream.readlines()
