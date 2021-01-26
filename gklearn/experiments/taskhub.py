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
         'file': 'Analysis_stability.ratios.real_data.relative_error.py'
		 },
		]

	command = ''
	for t in tasks:
		command += 'cd ' + t['path'] + '\n'
		command += 'python3 ' + t['file'] + '\n'
		command += 'cd ' + '/'.join(['..'] * len(t['path'].split('/'))) + '\n'

	import os
	os.system(command)