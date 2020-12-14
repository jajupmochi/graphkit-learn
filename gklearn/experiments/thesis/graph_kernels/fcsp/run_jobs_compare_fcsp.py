#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:49:43 2020

@author: ljia
"""

import os
import re


def get_job_script(param):
	script = r"""
#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="fcsp.""" + param + r""""
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_fcsp.""" + param + r""".txt"
#SBATCH --error="errors/error_fcsp.""" + param + r""".txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4000

srun hostname
srun cd /home/2019015/ljia02/graphkit-learn/gklearn/experiments/thesis/graph_kernels/fcsp
srun python3 compare_fcsp.py """ + param
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return script


if __name__ == '__main__':
	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	param_list = ['True', 'False']
	for param in param_list[:]:
		job_script = get_job_script(param)
		command = 'sbatch <<EOF\n' + job_script + '\nEOF'
# 			print(command)
		os.system(command)
	# 		os.popen(command)
	# 		output = stream.readlines()