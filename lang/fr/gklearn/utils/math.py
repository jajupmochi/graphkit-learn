#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:43:36 2020

@author: ljia
"""

def rounder(x, decimals):
	"""Round, where 5 is rounded up.

	Parameters
	----------
	x : float
		The number to be rounded.
	decimals : int
		Decimals to which ``x'' is rounded.

	Returns
	-------
	string
		The rounded number.
	"""
	x_strs = str(x).split('.')
	if len(x_strs) == 2:
		before = x_strs[0]
		after = x_strs[1]
		if len(after) > decimals:
			if int(after[decimals]) >= 5:
				after0s = ''
				for c in after:
					if c == '0':
						after0s += '0'
					elif c != '0':
						break
				if len(after0s) == decimals:
					after0s = after0s[:-1]
				after = after0s + str(int(after[0:decimals]) + 1)[-decimals:]
			else:
				after = after[0:decimals]
		elif len(after) < decimals:
			after += '0' * (decimals - len(after))
		return before + '.' + after

	elif len(x_strs) == 1:
		return x_strs[0]
	
	
if __name__ == '__main__':
	x = 1.0075333616
	y = rounder(x, 2)
	print(y)