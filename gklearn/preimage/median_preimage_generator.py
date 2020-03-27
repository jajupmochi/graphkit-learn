#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:27:22 2020

@author: ljia
"""
from gklearn.preimage.preimage_generator import PreimageGenerator
# from gklearn.utils.dataset import Dataset

class MedianPreimageGenerator(PreimageGenerator):
	
	def __init__(self, mge, dataset):
		self.__mge = mge
		self.__dataset = dataset