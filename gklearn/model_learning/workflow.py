#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:33:51 2020

@author: ljia
"""
import os
import numpy as np
import pickle
from gklearn.dataset import Dataset
from gklearn.model_learning import NestedCV
from gklearn.kernels import GRAPH_KERNELS

class Workflow(object):


	def __init__(self, **kwargs):
		self._job_prefix = kwargs.get('job_prefix', 'gktask')
		self._max_num_running_tasks = kwargs.get('max_num_running_tasks', np.inf)
		self._root_dir = kwargs.get('root_dir', 'outputs/')


	def run(self, tasks):
		### Check inputs.
		if self._check_inputs(tasks):
			self._tasks = tasks
		else:
			raise ValueError('The input "tasks" is not correct.')


		### Sort tasks.
		self.sort_tasks_by_complexity()


		### The main process.
		complete = False
		while not complete:

			self.get_running_tasks()

			if self._num_running_tasks < self._max_num_running_tasks:

				### Load results from table.
				self.load_results_from_table()

				for task in self._tasks:
					state = self.get_task_state(task)
					if state != 'complete' and state != 'runnning':
						self.run_task(task)

					if self._num_running_tasks >= self._max_num_running_tasks:
						break

			### Save results.
			self.save_results()

			complete = self.check_completeness()

# 			sleep()


	def _check_inputs(self, tasks):
		if not isinstance(tasks, list):
			return False
		else:
			for i in tasks:
				if not 'kernel' in i or not 'dataset' in i:
					return False
		return True


	def sort_tasks_by_complexity(self):
		return


	def get_running_tasks(self):
		command = 'squeue --user $USER --format "%.50j" --noheader'
		stream = os.popen(command)
		output = stream.readlines()
		running_tasks = [o for o in output if o.strip().startswith(self._job_prefix)]
		self._num_running_tasks = len(running_tasks)


	def load_results_from_table(self):
		pass


	def get_task_state(self, task):
		task_dir = os.path.join(self._root_dir, task['kernel'] + '.' + task['dataset'] + '/')
		fn_summary = os.path.join(task_dir, 'results_summary.pkl')
		if os.path.isfile(fn_summary):
			output = pickle.loads(fn_summary)
			state = output['state']
			return state
		else:
			return 'unstarted'


	def run_task(self, task):
		ds_name = task['dataset']
		k_name = task['kernel']

		# Get dataset.
		ds = Dataset(ds_name)
		graph_kernel = GRAPH_KERNELS[k_name]

		# Start CV.
		results = NestedCV(ds, graph_kernel)