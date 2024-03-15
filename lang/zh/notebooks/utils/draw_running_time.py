#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw running time.
Created on Mon Sep 24 17:37:26 2018

@author: ljia
"""

import numpy as np
import matplotlib.pyplot as plt

N = 7
tgm1 = np.array([0.73, 
         0.88,
         1.65,
         1.97,
         4.89,
         36.98,
         704.54])
tgm2 = np.array([0.77,
        1.22,
        2.95,
        5.70,
        20.29,
        147.09,
        3477.65])
tms1 = np.array([2.68, 
        3.41, 
        3.36,
        237.00,
        7.58,
        255.48,
        717.35])
tms2 = np.array([3.93, 
        4.96, 
        5.84,
        833.06,
        26.62,
        807.84,
        3515.72])

fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.2))

ind = np.arange(N)    # the x locations for the groups
width = 0.23       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind - width * 0.03, tgm1, width, label='compute Gram matrix on $CRIANN$ ($t_1$)', zorder=3)
p2 = ax.bar(ind - width * 0.03, tms1 - tgm1, width, bottom=tgm1, label='model selection on $CRIANN$', zorder=3)
p3 = ax.bar(ind + width * 1.03, tgm2, width, label='compute Gram matrix on $laptop$ ($t_2$)', zorder=3)
p4 = ax.bar(ind + width * 1.03, tms2 - tgm2, width, bottom=tgm2, label='model selection on $laptop$', zorder=3)

ax.set_yscale('log', nonposy='clip')
ax.set_xlabel('datasets')
ax.set_ylabel('runtime($s$)')
#ax.set_title('Runtime of the shortest path kernel on all datasets')
plt.xticks(ind + width / 2, ('Alkane', 'Acyclic', 'MAO', 'PAH', 'MUTAG', 
                             'Letter-med', 'ENZYMES'))
#ax.set_yticks(np.logspace(-16, -3, num=20, base=10))
#ax.set_ylim(bottom=1e-15)
ax.grid(axis='y', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('none')

ax2 = ax.twinx()
p5 = ax2.plot(ind + width / 2, tgm2 / tgm1, 'bo-', 
              label='$t_2 / $ $t_1$')
ax2.set_ylabel('ratios')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_ticks_position('none')
ax2.yaxis.set_ticks_position('none'
                             )
ax.yaxis.set_ticks_position('none')

fig.subplots_adjust(right=0.63)
fig.legend(loc='right', ncol=1, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)

plt.savefig('../check_gm/parallel_runtime_on_different_machines.eps', format='eps', dpi=300,
            transparent=True, bbox_inches='tight')
plt.show()