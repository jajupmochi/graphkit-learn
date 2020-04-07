#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:38:59 2020

@author: ljia
"""

from tqdm import tqdm
import sys

print('start')

for i in tqdm(range(10000000), file=sys.stdout):
 	x = i
# 	print(x)
# =============================================================================
# summary
# terminal, IPython 7.0.1 (Spyder 4): Works.
# write to file: does not work. Progress bar splits as the progress goes.
# Jupyter:
# =============================================================================

# for i in tqdm(range(10000000)):
# 	x = i
# 	print(x)
# =============================================================================
# summary 
# terminal, IPython 7.0.1 (Spyder 4): does not work. When combines with other
# print, progress bar splits.
# write to file: does not work. Cannot write progress bar to file.
# Jupyter:
# =============================================================================
