#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:52:50 2020

@author: ljia
"""
import time

class Timer(object):
	"""A timer class that can be used by methods that support time limits.
	
	Note
	----
	This is the Python implementation of `the C++ code in GEDLIB <https://github.com/dbblumenthal/gedlib/blob/master/src/env/timer.hpp>`__.
	"""
	
	def __init__(self, time_limit_in_sec):
		"""Constructs a timer for a given time limit.
		
		Parameters
		----------
		time_limit_in_sec : string
			The time limit in seconds.
		"""		
		self._time_limit_in_sec = time_limit_in_sec
		self._start_time = time.time()
	
	
	def expired(self):
		"""Checks if the time limit has expired. 
		
		Return
		------
		Boolean true if the time limit has expired and false otherwise.
"""
		if self._time_limit_in_sec > 0:
			runtime = time.time() - self._start_time
			return runtime >= self._time_limit_in_sec
		return False