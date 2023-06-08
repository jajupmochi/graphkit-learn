#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:02:17 2022

@author: ljia
"""
import sys
from gklearn.ged.model.distances import euclid_d
from gklearn.ged.util import pairwise_ged, get_nb_edit_operations
from gklearn.utils import get_iters


def compute_ged(
		Gi, Gj, edit_cost, edit_cost_fun='CONSTANT', method='BIPARTITE',
		repeats=10, **kwargs
):
	"""
	Compute GED between two graph according to edit_cost.
	"""
	ged_options = {
		'edit_cost': edit_cost_fun,
		'method': method,
		'edit_cost_constants': edit_cost
	}
	node_labels = kwargs.get('node_labels', [])
	edge_labels = kwargs.get('edge_labels', [])
	dis, pi_forward, pi_backward = pairwise_ged(
		Gi, Gj, ged_options, repeats=repeats
	)
	n_eo_tmp = get_nb_edit_operations(
		Gi, Gj, pi_forward, pi_backward, edit_cost=edit_cost_fun,
		node_labels=node_labels, edge_labels=edge_labels
	)
	return dis, n_eo_tmp


def compute_ged_all_dataset(Gn, edit_cost, ed_method, **kwargs):
	N = len(Gn)
	G_pairs = []
	for i in range(N):
		for j in range(i, N):
			G_pairs.append([i, j])
	return compute_geds(G_pairs, Gn, edit_cost, ed_method, **kwargs)


def compute_geds(
		G_pairs, Gn, edit_cost, ed_method, edit_cost_fun='CONSTANT',
		verbose=True, **kwargs
):
	"""
	Compute GED between all indexes in G_pairs given edit_cost
	:return: ged_vec : the list of computed distances, n_edit_operations : the list of edit operations
	"""
	ged_vec = []
	n_edit_operations = []
	for k in get_iters(
			range(len(G_pairs)), desc='Computing GED', file=sys.stdout,
			length=len(G_pairs), verbose=verbose
	):
		[i, j] = G_pairs[k]
		dis, n_eo_tmp = compute_ged(
			Gn[i], Gn[j], edit_cost=edit_cost, edit_cost_fun=edit_cost_fun,
			method=ed_method, **kwargs
		)
		ged_vec.append(dis)
		n_edit_operations.append(n_eo_tmp)

	return ged_vec, n_edit_operations


def compute_D(G_app, edit_cost, G_test=None, ed_method='BIPARTITE', **kwargs):
	import numpy as np
	N = len(G_app)
	D_app = np.zeros((N, N))

	for i, G1 in get_iters(
			enumerate(G_app), desc='Computing D - app', file=sys.stdout,
			length=N
	):
		for j, G2 in enumerate(G_app[i + 1:], i + 1):
			D_app[i, j], _ = compute_ged(
				G1, G2, edit_cost, method=ed_method, **kwargs
			)
			D_app[j, i] = D_app[i, j]
	if (G_test is None):
		return D_app, edit_cost
	else:
		D_test = np.zeros((len(G_test), N))
		for i, G1 in get_iters(
				enumerate(G_test), desc='Computing D - test', file=sys.stdout,
				length=len(G_test)
		):
			for j, G2 in enumerate(G_app):
				D_test[i, j], _ = compute_ged(
					G1, G2, edit_cost, method=ed_method, **kwargs
				)
		return D_app, D_test, edit_cost


def compute_D_random(G_app, G_test=None, ed_method='BIPARTITE', **kwargs):
	import numpy as np
	edit_costs = np.random.rand(6)
	return compute_D(G_app, edit_costs, G_test, ed_method=ed_method, **kwargs)


def compute_D_expert(G_app, G_test=None, ed_method='BIPARTITE', **kwargs):
	edit_cost = [3, 3, 1, 3, 3, 1]
	return compute_D(G_app, edit_cost, G_test, ed_method=ed_method, **kwargs)


def compute_D_fitted(
		G_app, y_app, G_test=None, y_distance=euclid_d,
		mode='reg', unlabeled=False, ed_method='BIPARTITE', **kwargs
):
	from gklearn.ged.models.optim_costs import compute_optimal_costs

	costs_optim = compute_optimal_costs(
		G_app, y_app, y_distance=y_distance,
		mode=mode, unlabeled=unlabeled, ed_method=ed_method, **kwargs
	)
	return compute_D(G_app, costs_optim, G_test, ed_method=ed_method, **kwargs)


def compute_D_GH2020(G_app, G_test=None, ed_method='BIPARTITE', **kwargs):
	from gklearn.ged.optim_costs import get_optimal_costs_GH2020
	costs_optim = get_optimal_costs_GH2020(**kwargs)
	return compute_D(G_app, costs_optim, G_test, ed_method=ed_method, **kwargs)
