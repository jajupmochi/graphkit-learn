from distances import euclid_d


def split_data(D, y, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	return D_app, D_test, y_app, y_test


def evaluate_D(D_app, y_app, D_test, y_test, mode='reg'):
	from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
	from distances import rmse, accuracy
	from sklearn.model_selection import GridSearchCV

	if (mode == 'reg'):
		knn = KNeighborsRegressor(metric='precomputed')
		scoring = 'neg_root_mean_squared_error'
		perf_eval = rmse
	else:
		knn = KNeighborsClassifier(metric='precomputed')
		scoring = 'accuracy'
		perf_eval = accuracy
	grid_params = {
		'n_neighbors': [3, 5, 7, 9, 11]
	}

	clf = GridSearchCV(knn, param_grid=grid_params,
					   scoring=scoring,
					   cv=5, return_train_score=True, refit=True)
	clf.fit(D_app, y_app)
	y_pred_app = clf.predict(D_app)
	y_pred_test = clf.predict(D_test)
	return perf_eval(y_pred_app, y_app), perf_eval(y_pred_test, y_test), clf


def xp_knn(Gn, y_all, y_distance=euclid_d,
		   mode='reg', unlabeled=False, ed_method='BIPARTITE', **kwargs):
	'''
	Perform a knn regressor on given dataset
	'''
	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
	from ged import compute_D_random, compute_D_expert
	from ged import compute_D_fitted

	stratified = False
	if mode == 'classif':
		stratified = True

	if stratified:
		rs = StratifiedShuffleSplit(n_splits=10, test_size=.1)
	else:
		rs = ShuffleSplit(n_splits=10, test_size=.1)

	if stratified:
		split_scheme = rs.split(Gn, y_all)
	else:
		split_scheme = rs.split(Gn)

	results = []
	i = 1
	for train_index, test_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, 10))
		i = i + 1
		cur_results = {}
		# Get splitted data
		G_app, G_test, y_app, y_test = split_data(Gn, y_all,
												  train_index, test_index)

		cur_results['y_app'] = y_app
		cur_results['y_test'] = y_test

		# Feed distances will all methods to compare
		distances = {}
		distances['random'] = compute_D_random(G_app, G_test, ed_method, **kwargs)
		distances['expert'] = compute_D_expert(G_app, G_test, ed_method, **kwargs)
		distances['fitted'] = compute_D_fitted(
			G_app, y_app, G_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			**kwargs)

		for setup in distances.keys():
			print("{0} Mode".format(setup))
			setup_results = {}
			D_app, D_test, edit_costs = distances[setup]
			setup_results['D_app'] = D_app
			setup_results['D_test'] = D_test
			setup_results['edit_costs'] = edit_costs
			print(edit_costs)
			perf_app, perf_test, clf = evaluate_D(
				D_app, y_app, D_test, y_test, mode)

			setup_results['perf_app'] = perf_app
			setup_results['perf_test'] = perf_test
			setup_results['clf'] = clf

			print(
				"Learning performance with {1} costs : {0:.2f}".format(
					perf_app, setup))
			print(
				"Test performance with {1} costs : {0:.2f}".format(
					perf_test, setup))
			cur_results[setup] = setup_results
		results.append(cur_results)
	return results
