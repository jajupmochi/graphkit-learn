#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:55:17 2020

@author: ljia

@references: 

	[1] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

from gklearn.kernels import SylvesterEquation, ConjugateGradient, FixedPoint, SpectralDecomposition


class RandomWalk(SylvesterEquation, ConjugateGradient, FixedPoint, SpectralDecomposition):
	
	
	def __init__(self, **kwargs):
		self._compute_method = kwargs.get('compute_method', None)
		self._compute_method = self._compute_method.lower()
		
		if self._compute_method == 'sylvester':
			self._parent = SylvesterEquation
		elif self._compute_method == 'conjugate':
			self._parent = ConjugateGradient
		elif self._compute_method == 'fp':
			self._parent = FixedPoint
		elif self._compute_method == 'spectral':
			self._parent = SpectralDecomposition
		elif self._compute_method == 'kon':
			raise Exception('This computing method is not completed yet.')
		else:
			raise Exception('This computing method does not exist. The possible choices inlcude: "sylvester", "conjugate", "fp", "spectral".')

		self._parent.__init__(self, **kwargs)
		
		
	def _compute_gm_series(self):
		return self._parent._compute_gm_series(self)


	def _compute_gm_imap_unordered(self):
		return self._parent._compute_gm_imap_unordered(self)
	
		
	def _compute_kernel_list_series(self, g1, g_list):
		return self._parent._compute_kernel_list_series(self, g1, g_list)

	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		return self._parent._compute_kernel_list_imap_unordered(self, g1, g_list)
	
	
	def _compute_single_kernel_series(self, g1, g2):
		return self._parent._compute_single_kernel_series(self, g1, g2)