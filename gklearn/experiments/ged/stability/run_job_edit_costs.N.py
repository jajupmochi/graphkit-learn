#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:23:25 2020

@author: ljia
"""
import os
import re


def get_job_script(arg, params):
	ged_method = params[0]
	multi_method = params[1]
	job_name_label = r"rep." if multi_method == 'repeats' else r""
	script = r"""
#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="st.""" + job_name_label + r"N" + arg + r"." + ged_method + r""""
#SBATCH --partition=tlong
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_edit_costs.""" + multi_method + r".N." + ged_method + r"." + arg + r""".txt"
#SBATCH --error="errors/error_edit_costs.""" + multi_method + r".N." + ged_method + r"." + arg + r""".txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=300:00:00
#SBATCH --mem-per-cpu=4000

srun hostname
srun cd /home/2019015/ljia02/graphkit-learn/gklearn/experiments/ged/stability
srun python3 edit_costs.""" + multi_method + r".N." + ged_method + r".py " + arg
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)
	
	return script

if __name__ == '__main__':
	
	params_list = [('IPFP', 'nums_sols'), 
				   ('IPFP', 'repeats'), 
				   ('bipartite', 'max_num_sols'), 
				   ('bipartite', 'repeats')]
	N_list = [10, 50, 100]
	for params in params_list[1:]:
		for N in [N_list[i] for i in [0, 1, 2]]:
			job_script = get_job_script(str(N), params)
			command = 'sbatch <<EOF\n' + job_script + '\nEOF'
# 			print(command)
			os.system(command)
	# 		os.popen(command)
	# 		output = stream.readlines()