#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw running time.
Created on Mon Sep 24 17:37:26 2018

@author: ljia
"""

import numpy as np
import matplotlib.pyplot as plt

N = 6
tgm1 = [3.68, 
         2.24,
         3.34,
#         0,
         20.00,
         2020.46,
         3198.84]
tgm2 = [4.29,
         3.35,
         5.78,
#         11.21,
         40.58,
         3136.26,
         17222.21]
tms1 = [51.19, 
          73.09, 
          5.01,
#          0,
          22.87,
          2211.97,
          3211.58]
tms2 = [65.16, 
          53.02, 
          10.32,
#          1162.41,
          49.86,
          3931.68,
          17270.55]

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.30       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, tgm1, width, label='$t_{gm}$ CRIANN')
p2 = ax.bar(ind, tms1, width, bottom=tgm1, label='$t_{ms}$ CRIANN')
p3 = ax.bar(ind + width, tgm2, width, label='$t_{gm}$ laptop')
p4 = ax.bar(ind + width, tms2, width, bottom=tgm2, label='$t_{ms}$ laptop')

ax.set_yscale('log', nonposy='clip')
ax.set_xlabel('datasets')
ax.set_ylabel('runtime($s$)')
ax.set_title('Runtime of the shortest path kernel on all datasets')
plt.xticks(ind + width / 2, ('Acyclic', 'Alkane', 'MAO', 'MUTAG', 'Letter-med', 'ENZYMES'))
#ax.set_yticks(np.logspace(-16, -3, num=20, base=10))
#ax.set_ylim(bottom=1e-15)
ax.legend(loc='upper left')

ax2 = ax.twinx()
p1 = ax2.plot(ind + width / 2, np.array(tgm2) / np.array(tgm1), 'ro-', 
              label='$t_{gm}$ laptop / $t_{gm}$ CRIANN')
ax2.set_ylabel('ratios')
ax2.legend(loc='upper center')

plt.savefig('check_gm/compare_running_time.eps', format='eps', dpi=300)
plt.show()