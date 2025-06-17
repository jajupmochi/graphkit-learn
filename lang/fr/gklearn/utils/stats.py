#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:12:41 2020

@author: ljia
"""
from collections import Counter
from scipy import stats


def entropy(labels, base=None):
	"""Calculate the entropy of a distribution for given list of labels.

	Parameters
	----------
	labels : list
		Given list of labels.
	base : float, optional
		The logarithmic base to use. The default is ``e`` (natural logarithm).

	Returns
	-------
	float
		The calculated entropy.
	"""
	return stats.entropy(list(Counter(labels).values()), base=base)