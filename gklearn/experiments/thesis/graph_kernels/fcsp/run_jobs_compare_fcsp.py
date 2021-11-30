#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:49:43 2020

@author: ljia
"""

import os
import re


OUT_TIME_LIST = set({('ShortestPath', 'ENZYMES', 'False'),
				 ('StructuralSP', 'ENZYMES', 'True'),
				 ('StructuralSP', 'ENZYMES', 'False'),
				 ('StructuralSP', 'AIDS', 'False'),
				 ('ShortestPath', 'NCI1', 'False'),
				 ('StructuralSP', 'NCI1', 'True'),
				 ('StructuralSP', 'NCI1', 'False'),
				 ('ShortestPath', 'NCI109', 'False'),
				 ('StructuralSP', 'NCI109', 'True'),
				 ('ShortestPath', 'NCI-H23', 'True'),
				 ('ShortestPath', 'NCI-H23', 'False'),
				 ('StructuralSP', 'NCI-H23', 'True'),
				 ('StructuralSP', 'NCI-H23', 'False'),
				 ('StructuralSP', 'NCI109', 'False'),
				 ('ShortestPath', 'NCI-H23H', 'True'),
				 ('ShortestPath', 'NCI-H23H', 'False'),
				 ('StructuralSP', 'NCI-H23H', 'True'),
				 ('StructuralSP', 'NCI-H23H', 'False'),
				 ('ShortestPath', 'DD', 'True'),
				 ('ShortestPath', 'DD', 'False'),
				 ('StructuralSP', 'BZR', 'False'),
				 ('ShortestPath', 'COX2', 'False'),
				 ('StructuralSP', 'COX2', 'False'),
				 ('ShortestPath', 'DHFR', 'False'),
				 ('StructuralSP', 'DHFR', 'False'),
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
				 ('StructuralSP', 'OHSU', 'True'),
				 ('StructuralSP', 'OHSU', 'False'),
				 ('ShortestPath', 'OVCAR-8', 'True'),
				 ('ShortestPath', 'OVCAR-8', 'False'),
				 ('StructuralSP', 'OVCAR-8', 'True'),
				 ('StructuralSP', 'OVCAR-8', 'False'),
				 ('ShortestPath', 'OVCAR-8H', 'True'),
				 ('ShortestPath', 'OVCAR-8H', 'False'),
				 ('StructuralSP', 'OVCAR-8H', 'True'),
				 ('StructuralSP', 'OVCAR-8H', 'False'),
				 ('ShortestPath', 'P388', 'False'),
				 ('ShortestPath', 'P388', 'True'),
				 ('StructuralSP', 'P388', 'True'),
				 ('StructuralSP', 'Steroid', 'False'),
				 ('ShortestPath', 'SYNTHETIC', 'False'),
				 ('StructuralSP', 'SYNTHETIC', 'True'),
				 ('StructuralSP', 'SYNTHETIC', 'False'),
				 ('ShortestPath', 'SYNTHETICnew', 'False'),
				 ('StructuralSP', 'SYNTHETICnew', 'True'),
				 ('StructuralSP', 'SYNTHETICnew', 'False'),
				 ('ShortestPath', 'Synthie', 'False'),
				 ('StructuralSP', 'Synthie', 'True'),
				 ('StructuralSP', 'Synthie', 'False'),
				 ('ShortestPath', 'COIL-DEL', 'False'),
				 ('StructuralSP', 'COIL-DEL', 'True'),
				 ('StructuralSP', 'COIL-DEL', 'False'),
				 ('ShortestPath', 'PROTEINS', 'False'),
				 ('ShortestPath', 'PROTEINS_full', 'False'),
				 ('StructuralSP', 'Mutagenicity', 'True'),
				 ('StructuralSP', 'Mutagenicity', 'False'),
				 ('StructuralSP', 'REDDIT-BINARY', 'True'),
				 ('StructuralSP', 'REDDIT-BINARY', 'False'),
				 ('StructuralSP', 'Vitamin_D', 'False'),
				 ('ShortestPath', 'Web', 'True'),
				 ('ShortestPath', 'Web', 'False'),
				 })

OUT_MEM_LIST = set({('StructuralSP', 'DD', 'True'),
					('StructuralSP', 'DD', 'False'),
					('StructuralSP', 'PROTEINS', 'True'),
					('StructuralSP', 'PROTEINS', 'False'),
					('StructuralSP', 'PROTEINS_full', 'True'),
					('StructuralSP', 'PROTEINS_full', 'False'),
					('ShortestPath', 'REDDIT-BINARY', 'True'),
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
	script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name="fcsp.""" + kernel + r"." + dataset + r"." + fcsp + r""""
#SBATCH --partition=tlong
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_fcsp.""" + kernel + r"." + dataset + r"." + fcsp + r""".txt"
#SBATCH --error="errors/error_fcsp.""" + kernel + r"." + dataset + r"." + fcsp + r""".txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=300:00:00
##SBATCH --mem-per-cpu=4000
#SBATCH --mem=40000

srun hostname
srun cd /home/2019015/ljia02/graphkit-learn/gklearn/experiments/thesis/graph_kernels/fcsp
srun python3 compare_fcsp.py """ + kernel + r" " + dataset + r" " + fcsp
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
	command = 'squeue --user $USER --name "fcsp' + str_task_id + '" --format "%.2t" --noheader'
	stream = os.popen(command)
	output = stream.readlines()
	if len(output) > 0:
		return True

	# Check if there are more than 10 tlong tasks running.
	command = 'squeue --user $USER --partition tlong --noheader'
	stream = os.popen(command)
	output = stream.readlines()
	if len(output) >= 10:
		return True


	# Check if the results are already computed.
	file_name = os.path.join(save_dir, 'run_time' + str_task_id + '.pkl')
	if os.path.isfile(file_name):
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
