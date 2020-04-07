#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:26:36 2020

@author: ljia
"""
# from gklearn.utils import Dataset

class PreimageGenerator(object):
	
	def __init__(self, dataset=None):
		# arguments to set.
		self._dataset = None if dataset is None else dataset
		self._kernel_options = {}
		self._graph_kernel = None
		self._verbose = 2

	@property
	def dataset(self):
		return self._dataset

	@dataset.setter
	def dataset(self, value):
		self._dataset = value		
		
	
	@property
	def kernel_options(self):
		return self._kernel_options
	
	@kernel_options.setter
	def kernel_options(self, value):
		self._kernel_options = value


	@property
	def graph_kernel(self):
		return self._graph_kernel
		
		
	@property
	def verbose(self):
		return self._verbose

	@verbose.setter
	def verbose(self, value):
		self._verbose = value

