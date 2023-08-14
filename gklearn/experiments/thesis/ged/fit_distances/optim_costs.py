from ged import compute_geds
from distances import sum_squares, euclid_d
import numpy as np
# from tqdm import tqdm

import sys


# sys.path.insert(0, "../")


def optimize_costs_unlabeled(nb_cost_mat, dis_k_vec):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
	:param dis_k_vec: The N distances to fit
	"""
	import cvxpy as cp
	import numpy as np
	MAX_SAMPLE = 1000
	nb_cost_mat_m = np.array([[x[0], x[1], x[3], x[4]] for x in nb_cost_mat])
	dis_k_vec = np.array(dis_k_vec)
	# dis_k_vec_norm = dis_k_vec/np.max(dis_k_vec)

	# import pickle
	# pickle.dump([nb_cost_mat, dis_k_vec], open('debug', 'wb'))
	N = nb_cost_mat_m.shape[0]
	sub_sample = np.random.permutation(np.arange(N))
	sub_sample = sub_sample[:MAX_SAMPLE]

	x = cp.Variable(nb_cost_mat_m.shape[1])
	cost = cp.sum_squares(
		(nb_cost_mat_m[sub_sample, :] @ x) - dis_k_vec[sub_sample]
	)
	prob = cp.Problem(cp.Minimize(cost), [x >= 0])
	prob.solve()
	edit_costs_new = [x.value[0], x.value[1], 0, x.value[2], x.value[3], 0]
	edit_costs_new = [xi if xi > 0 else 0 for xi in edit_costs_new]
	residual = prob.value
	return edit_costs_new, residual


def optimize_costs_classif_unlabeled(nb_cost_mat, Y):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in
	nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit
	operations for each pair of graph
	:param dis_k_vec: {-1,1}^N vector of common classes
	"""
	# import cvxpy as cp
	from ml import reg_log
	# import pickle
	# pickle.dump([nb_cost_mat, Y], open('debug', 'wb'))
	nb_cost_mat_m = np.array(
		[[x[0], x[1], x[3], x[4]]
		 for x in nb_cost_mat]
	)
	w, J, _ = reg_log(nb_cost_mat_m, Y, pos_contraint=True)
	edit_costs_new = [w[0], w[1], 0, w[2], w[3], 0]
	residual = J[-1]

	return edit_costs_new, residual


def optimize_costs_classif(nb_cost_mat, Y):
	"""
		Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
		! take care that nb_cost_mat do not contains 0 lines
		:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
		:param dis_k_vec: {-1,1}^N vector of common classes
	"""
	# import pickle
	# pickle.dump([nb_cost_mat, Y], open("test.pickle", "wb"))
	from ml import reg_log
	w, J, _ = reg_log(nb_cost_mat, Y, pos_contraint=True)
	return w, J[-1]


def optimize_costs(nb_cost_mat, dis_k_vec):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
	:param dis_k_vec: The N distances to fit
	"""
	import cvxpy as cp
	x = cp.Variable(nb_cost_mat.shape[1])
	cost = cp.sum_squares((nb_cost_mat @ x) - dis_k_vec)
	constraints = [
		x >= [0.01 for i in range(nb_cost_mat.shape[1])],
		np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T @ x >= 0.0,
		np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T @ x >= 0.0
	]
	prob = cp.Problem(cp.Minimize(cost), constraints)
	prob.solve()
	edit_costs_new = x.value
	residual = prob.value

	return edit_costs_new, residual


def compute_optimal_costs(
		G, y, init_costs=[3, 3, 1, 3, 3, 1],
		y_distance=euclid_d,
		mode='reg', unlabeled=False,
		ed_method='BIPARTITE',
		**kwargs
):
	N = len(y)

	G_pairs = []
	distances_vec = []

	for i in range(N):
		for j in range(i + 1, N):
			G_pairs.append([i, j])
			distances_vec.append(y_distance(y[i], y[j]))
	ged_vec_init, n_edit_operations = compute_geds(
		G_pairs, G, init_costs, ed_method, **kwargs
	)

	residual_list = [sum_squares(ged_vec_init, distances_vec)]

	if (mode == 'reg'):
		if unlabeled:
			method_optim = optimize_costs_unlabeled
		else:
			method_optim = optimize_costs

	elif (mode == 'classif'):
		if unlabeled:
			method_optim = optimize_costs_classif_unlabeled
		else:
			method_optim = optimize_costs_classif

	ite_max = 5
	for i in range(ite_max):
		print('ite', i + 1, '/', ite_max, ':')
		# compute GEDs and numbers of edit operations.
		edit_costs_new, residual = method_optim(
			np.array(n_edit_operations), distances_vec
		)
		ged_vec, n_edit_operations = compute_geds(
			G_pairs, G, edit_costs_new, ed_method, **kwargs
		)
		residual_list.append(sum_squares(ged_vec, distances_vec))

	return edit_costs_new
