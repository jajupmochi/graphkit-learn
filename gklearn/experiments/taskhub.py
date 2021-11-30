#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:53:33 2021

@author: ljia
"""

if __name__ == '__main__':
	tasks = [
		{'path': 'thesis/graph_kernels/fcsp',
         'file': 'run_jobs_compare_fcsp.py'
		 },
		{'path': 'thesis/graph_kernels/fcsp',
         'file': 'run_jobs_compare_fcsp_space.py'
		 },
		{'path': 'ged/stability',
         'file': 'run_job_edit_costs.real_data.nums_sols.ratios.IPFP.py'
		 },
		]

	import os
	for t in tasks:
		print(t['file'])
		command = ''
		command += 'cd ' + t['path'] + '\n'
		command += 'python3 ' + t['file'] + '\n'
# 		command += 'cd ' + '/'.join(['..'] * len(t['path'].split('/'))) + '\n'
		os.system(command)
