
import sys


def run_xp(ds_name, output_file, unlabeled, mode, y_distance, ed_method):
	from gklearn.dataset import Dataset
	from gklearn.experiments import DATASET_ROOT
	from learning import xp_knn

	ds = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
	ds.remove_labels(node_attrs=ds.node_attrs, edge_attrs=ds.edge_attrs) # @todo: ged can not deal with sym and unsym labels.
	Gn = ds.graphs
	y_all = ds.targets

	resu = {}
	resu['y_distance'] = y_distance
	resu['dataset'] = ds_name
	unlabeled = (len(ds.node_labels) == 0 and len(ds.edge_labels) == 0)
	results = xp_knn(Gn, y_all, y_distance=y_distances[y_distance],
				  mode=mode,
				  unlabeled=unlabeled, ed_method=ed_method,
				  node_labels=ds.node_labels, edge_labels=ds.edge_labels)
	resu['results'] = results
	resu['unlabeled'] = unlabeled
	resu['mode'] = mode
	resu['ed_method'] = ed_method
	pickle.dump(resu, open(output_result, 'wb'))
	return output_result


def run_from_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="path to / name of the dataset to predict")
	parser.add_argument(
		"output_file", help="path to file which will contains the results")
	parser.add_argument("-u", "--unlabeled", help="Specify that the dataset is unlabeled graphs",
						action="store_true")
	parser.add_argument("-m", "--mode", type=str, choices=['reg', 'classif'],
						help="Specify if the dataset a classification or regression problem")
	parser.add_argument("-y", "--y_distance", type=str, choices=['euclidean', 'manhattan', 'classif'],
						default='euclid',
						help="Specify the distance on y  to fit the costs")

	args = parser.parse_args()

	dataset = args.dataset
	output_result = args.output_file
	unlabeled = args.unlabeled
	mode = args.mode

	print(args)
	y_distances = {
		'euclidean': euclid_d,
		'manhattan': man_d,
		'classif': classif_d
	}
	y_distance = y_distances['euclid']

	run_xp(dataset, output_result, unlabeled, mode, y_distance)
	print("Fini")


if __name__ == "__main__":

	import pickle
	import os

	from distances import euclid_d, man_d, classif_d
	y_distances = {
		'euclidean': euclid_d,
		'manhattan': man_d,
		'classif': classif_d
	}

	# Read arguments.
	if len(sys.argv) > 1:
		run_from_args()
	else:
		from sklearn.model_selection import ParameterGrid

		# Get task grid.
		Edit_Cost_List = ['BIPARTITE', 'IPFP']
		Dataset_list = ['Alkane_unlabeled', 'Acyclic', 'Chiral', 'Vitamin_D',
					    'Steroid']
		Dis_List = ['euclidean', 'manhattan']
		task_grid = ParameterGrid({'edit_cost': Edit_Cost_List[0:1],
							 'dataset': Dataset_list[1:2],
							 'distance': Dis_List[:]})

		unlabeled = False # @todo: Not actually used.
		mode = 'reg'
		# Run.
		for task in list(task_grid):
			print()
			print(task)

			output_result = 'outputs/results.' + '.'.join([task['dataset'], task['edit_cost'], task['distance']]) + '.pkl'
			if not os.path.isfile(output_result):
				run_xp(task['dataset'], output_result, unlabeled, mode, task['distance'], task['edit_cost'])