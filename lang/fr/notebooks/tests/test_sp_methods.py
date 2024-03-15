#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test networkx shortest paths methods.
Created on Tue Oct  9 14:49:09 2018

@author: ljia
"""

import networkx as nx

g = nx.Graph()
g.add_edge(1, 2)
g.add_edge(3, 2)
g.add_edge(1, 4)
g.add_edge(3, 4)
p1 = nx.shortest_path(g, 1, 3)
p1 = [p1]
p2 = list(nx.all_shortest_paths(g, 1, 3))
p1 += p2
pr = [sp[::-1] for sp in p1]
nx.draw(g)