#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare gram matrices
Created on Mon Sep 24 10:52:25 2018

@author: ljia
"""
import numpy as np
import matplotlib.pyplot as plt

N = 7
egmin = [-3.425001366427846e-15, 
         -5.513191435356332e-15,
         -1.1563146193980238e-15,
         -1.3999833987183273e-15,
         -5.811474553224136e-15,
         2.196833029054622e-10,
         0.0001002381061317695]
egmin = np.abs(egmin)
egmin2 = [-9.433792343294819e-15, 
          np.NaN, 
          -7.502900269338164e-16,
          -1.3999833987183273e-15,
          -8.73626400337456e-15,
          np.NaN,
          -4.04460628433013e-14]
egmin2 = np.abs(egmin2)
egmax = [142.86649135778595,
         140.08307372708344,
         64.31844814063015,
         92.38382991977493,
         160.72585558445357,
         943.9175660197347,
         299.17895175532897]
egmax2 = [172.4203026547106, 
          np.NaN, 
          65.53092059526354,
          92.38382991977493,
          180.374192331094,
          np.NaN,
          529.3691973508182]

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.20       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, egmin, width)
p2 = ax.bar(ind, egmax, width, bottom=egmin)
p3 = ax.bar(ind + width, egmin2, width)
p4 = ax.bar(ind + width, egmax2, width, bottom=egmin2)

ax.set_yscale('log', nonposy='clip')
ax.set_xlabel('datasets')
ax.set_ylabel('absolute eigen values')
ax.set_title('Absolute eigen values of gram matrices on all datasets')
plt.xticks(ind + width / 2, ('Acyclic', 'Alkane', 'MAO', 'PAH', 'MUTAG', 'Letter-med', 'ENZYMES'))
#ax.set_yticks(np.logspace(-16, -3, num=20, base=10))
ax.set_ylim(bottom=1e-15)
ax.legend((p1[0], p2[0], p3[0], p4[0]), ('min1', 'max1', 'min2', 'max2'), loc='upper right')

plt.savefig('../check_gm/compare_eigen_values.eps', format='eps', dpi=300)
plt.show()