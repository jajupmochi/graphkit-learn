#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:18:02 2021

@author: ljia
"""

def dichotomous_permutation(arr, layer=0):
	import math

# 	def seperate_arr(arr, new_arr):
# 		if (length % 2) == 0:
# 			half = int(length / 2)
# 			new_arr += [arr[half - 1], arr[half]]
# 			subarr1 = [arr[i] for i in range(1, half - 1)]
# 		else:
# 			half = math.floor(length / 2)
# 			new_arr.append(arr[half])
# 			subarr1 = [arr[i] for i in range(1, half)]
# 		subarr2 = [arr[i] for i in range(half + 1, length - 1)]
# 		subarrs = [subarr1, subarr2]
# 		return subarrs


	if layer == 0:
		length = len(arr)
		if length <= 2:
			return arr

		new_arr = [arr[0], arr[-1]]
		if (length % 2) == 0:
 			half = int(length / 2)
 			new_arr += [arr[half - 1], arr[half]]
 			subarr1 = [arr[i] for i in range(1, half - 1)]
		else:
 			half = math.floor(length / 2)
 			new_arr.append(arr[half])
 			subarr1 = [arr[i] for i in range(1, half)]
		subarr2 = [arr[i] for i in range(half + 1, length - 1)]
		subarrs = [subarr1, subarr2]
# 		subarrs = seperate_arr(arr, new_arr)
		new_arr += dichotomous_permutation(subarrs, layer=layer+1)

	else:
		new_arr = []
		subarrs = []
		for a in arr:
			length = len(a)
			if length <= 2:
				new_arr += a
			else:
# 				subarrs += seperate_arr(a, new_arr)
				if (length % 2) == 0:
 					half = int(length / 2)
 					new_arr += [a[half - 1], a[half]]
 					subarr1 = [a[i] for i in range(0, half - 1)]
				else:
 					half = math.floor(length / 2)
 					new_arr.append(a[half])
 					subarr1 = [a[i] for i in range(0, half)]
				subarr2 = [a[i] for i in range(half + 1, length)]
				subarrs += [subarr1, subarr2]

		if len(subarrs) > 0:
			new_arr += dichotomous_permutation(subarrs, layer=layer+1)

	return new_arr

# 	length = len(arr)
# 	if length <= 2:
# 		return arr

# 	new_arr = [arr[0], arr[-1]]
# 	if (length % 2) == 0:
# 		half = int(length / 2)
# 		new_arr += [arr[half - 1], arr[half]]
# 		subarr1 = [arr[i] for i in range(1, half - 1)]
# 	else:
# 		half = math.floor(length / 2)
# 		new_arr.append(arr[half])
# 		subarr1 = [arr[i] for i in range(1, half)]
# 	subarr2 = [arr[i] for i in range(half + 1, length - 1)]
# 	if len(subarr1) > 0:
# 		new_arr += dichotomous_permutation(subarr1)
# 	if len(subarr2) > 0:
# 		new_arr += dichotomous_permutation(subarr2)

# 	return new_arr