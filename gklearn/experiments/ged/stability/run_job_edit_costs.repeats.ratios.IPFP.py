#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:23:25 2020

@author: ljia
"""
import os
import re


def get_job_script(arg):
	script = r"""
#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="st.rep.""" + arg + r""".IPFP"
#SBATCH --partition=tlong
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_edit_costs.repeats.ratios.IPFP.""" + arg + """.txt"
#SBATCH --error="errors/error_edit_costs.repeats.ratios.IPFP.""" + arg + """.txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=300:00:00
#SBATCH --mem-per-cpu=4000

srun hostname
srun cd /home/2019015/ljia02/graphkit-learn/gklearn/experiments/ged/stability
srun python3 edit_costs.repeats.ratios.IPFP.py """ + arg
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)
	
	return script

if __name__ == '__main__':
	ds_list = ['MAO', 'Monoterpenoides', 'MUTAG', 'AIDS_symb']
	for ds_name in [ds_list[i] for i in [0, 3]]:
		job_script = get_job_script(ds_name)
		command = 'sbatch <<EOF\n' + job_script + '\nEOF'
# 		print(command)
		os.system(command)
# 		os.popen(command)
# 		output = stream.readlines()