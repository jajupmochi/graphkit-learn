#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:35:26 2020

@author: ljia
"""

from tqdm import tqdm
import math


def get_iters(iterable, desc=None, file=None, length=None, verbose=True, **kwargs):
	if verbose:
		if 'miniters' not in kwargs:
			if length is None:
				try:
					kwargs['miniters'] = math.ceil(len(iterable) / 100)
				except TypeError:
					raise
					kwargs['miniters'] = 100
			else:
				kwargs['miniters'] = math.ceil(length / 100)
		if 'maxinterval' not in kwargs:
			kwargs['maxinterval'] = 600
		return tqdm(iterable, desc=desc, file=file, **kwargs)
	else:
		return iterable



# class mytqdm(tqdm):


# 	def __init__(iterable=None, desc=None, total=None, leave=True,
#                  file=None, ncols=None, mininterval=0.1, maxinterval=10.0,
#                  miniters=None, ascii=None, disable=False, unit='it',
#                  unit_scale=False, dynamic_ncols=False, smoothing=0.3,
#                  bar_format=None, initial=0, position=None, postfix=None,
#                  unit_divisor=1000, write_bytes=None, lock_args=None,
#                  nrows=None,
#                  gui=False, **kwargs):
# 		if iterable is not None:
# 			miniters=math.ceil(len(iterable) / 100)
# 		maxinterval=600
# 		super().__init__(iterable=iterable, desc=desc, total=total, leave=leave,
#                  file=file, ncols=ncols, mininterval=mininterval, maxinterval=maxinterval,
#                  miniters=miniters, ascii=ascii, disable=disable, unit=unit,
#                  unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing,
#                  bar_format=bar_format, initial=initial, position=position, postfix=postfix,
#                  unit_divisor=unit_divisor, write_bytes=write_bytes, lock_args=lock_args,
#                  nrows=nrows,
#                  gui=gui, **kwargs)

# tqdm = mytqdm