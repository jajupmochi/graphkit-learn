#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:25:49 2020

@author:
	Paul Zanoncelli, paul.zanoncelli@ecole.ensicaen.fr
	Luc Brun luc.brun@ensicaen.fr
	Sebastien Bougleux sebastien.bougleux@unicaen.fr
	Benoit Gaüzère benoit.gauzere@insa-rouen.fr
	Linlin Jia linlin.jia@insa-rouen.fr
"""
import os
import os.path as osp
import tarfile
from zipfile import ZipFile
# from gklearn.utils.graphfiles import loadDataset
# import torch.nn.functional as F
import networkx as nx
# import torch
import random
import sys
# from lxml import etree
import re
# from tqdm import tqdm
from gklearn.dataset import DATABASES, DATASET_META


class DataFetcher(object):

	def __init__(self, name=None, root='datasets', reload=False, verbose=False):
		self._name = name
		self._root = root
		if not osp.exists(self._root):
			os.makedirs(self._root)
		self._reload = reload
		self._verbose = verbose
		# 		self.has_train_valid_test = {
		# 			"Coil_Del" : ('COIL-DEL/data/test.cxl','COIL-DEL/data/train.cxl','COIL-DEL/data/valid.cxl'),
		# 			"Coil_Rag" : ('COIL-RAG/data/test.cxl','COIL-RAG/data/train.cxl','COIL-RAG/data/valid.cxl'),
		# 			"Fingerprint" : ('Fingerprint/data/test.cxl','Fingerprint/data/train.cxl','Fingerprint/data/valid.cxl'),
		# # 			"Grec" : ('GREC/data/test.cxl','GREC/data/train.cxl','GREC/data/valid.cxl'),
		# 			"Letter" : {'HIGH' : ('Letter/HIGH/test.cxl','Letter/HIGH/train.cxl','Letter/HIGH/validation.cxl'),
		# 						'MED' : ('Letter/MED/test.cxl','Letter/MED/train.cxl','Letter/MED/validation.cxl'),
		# 						'LOW' : ('Letter/LOW/test.cxl','Letter/LOW/train.cxl','Letter/LOW/validation.cxl')
		# 					   },
		# 			"Mutagenicity" : ('Mutagenicity/data/test.cxl','Mutagenicity/data/train.cxl','Mutagenicity/data/validation.cxl'),
		# # 			"Pah" : ['PAH/testset_0.ds','PAH/trainset_0.ds'],
		# 			"Protein" : ('Protein/data/test.cxl','Protein/data/train.cxl','Protein/data/valid.cxl'),
		# # 			"Web" : ('Web/data/test.cxl','Web/data/train.cxl','Web/data/valid.cxl')
		# 		}

		if self._name is None:
			if self._verbose:
				print(
					'No dataset name entered. All possible datasets will be loaded.'
				)
			self._name, self._path = [], []
			for idx, ds_name in enumerate(DATASET_META):
				if self._verbose:
					print(
						str(idx + 1), '/', str(len(DATASET_META)), 'Fetching',
						ds_name, end='... '
					)
				self._name.append(ds_name)
				success = self.write_archive_file(ds_name)
				if success:
					self._path.append(self.open_files(ds_name))
				else:
					self._path.append(None)
				if self._verbose and self._path[
					-1] is not None and not self._reload:
					print('Fetched.')

			if self._verbose:
				print(
					'Finished.', str(sum(v is not None for v in self._path)),
					'of', str(len(self._path)),
					'datasets are successfully fetched.'
				)

		elif self._name not in DATASET_META:
			message = 'Invalid dataset name "' + self._name + '".'
			message += '\nAvailable datasets are as follows: \n\n'
			message += '\n'.join(ds for ds in sorted(DATASET_META))
			message += '\n\nFollowing special suffices can be added to the name:'
			message += '\n\n' + '\n'.join(['_unlabeled'])
			raise ValueError(message)
		else:
			self.write_archive_file(self._name)
			self._path = self.open_files(self._name)


	# 		self.max_for_letter = 0
	# 		if mode == 'Pytorch':
	# 			if self._name in self.data_to_use_in_datasets :
	# 				Gs,y = self.dataset
	# 				inputs,adjs,y = self.from_networkx_to_pytorch(Gs,y)
	# 				#print(inputs,adjs)
	# 				self.pytorch_dataset = inputs,adjs,y
	# 			elif self._name == "Pah":
	# 				self.pytorch_dataset = []
	# 				test,train = self.dataset
	# 				Gs_test,y_test = test
	# 				Gs_train,y_train = train
	# 				self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_test,y_test))
	# 				self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_train,y_train))
	# 			elif self._name in self.has_train_valid_test:
	# 				self.pytorch_dataset = []
	# 				#[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])
	# 				test,train,valid = self.dataset
	# 				Gs_test,y_test = test
	#
	# 				Gs_train,y_train = train
	# 				Gs_valid,y_valid = valid
	# 				self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_test,y_test))
	# 				self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_train,y_train))
	# 				self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_valid,y_valid))
	# 			#############
	# 			"""
	# 			for G in Gs :
	# 				for e in G.edges():
	# 					print(G[e[0]])
	# 			"""
	# 			##############

	def download_file(self, url):
		import urllib.request
		import urllib.error
		try:
			response = urllib.request.urlopen(url)
		except urllib.error.HTTPError:
			print(
				'"', url.split('/')[-1],
				'" is not available or incorrect http link.'
			)
			return
		except urllib.error.URLError:
			print('Network is unreachable.')
			return
		return response


	def write_archive_file(self, ds_name):
		path = osp.join(self._root, ds_name)
		# 		filename_dir = osp.join(path,filename)
		if not osp.exists(path) or self._reload:
			url = DATASET_META[ds_name]['url']
			response = self.download_file(url)
			if response is None:
				return False
			os.makedirs(path, exist_ok=True)
			with open(os.path.join(path, url.split('/')[-1]), 'wb') as outfile:
				outfile.write(response.read())

		return True


	def open_files(self, ds_name=None):
		if ds_name is None:
			ds_name = (
				self._name if isinstance(self._name, str) else self._name[0])
		filename = DATASET_META[ds_name]['url'].split('/')[-1]
		path = osp.join(self._root, ds_name)
		filename_archive = osp.join(path, filename)

		if filename.endswith('gz'):
			if tarfile.is_tarfile(filename_archive):
				with tarfile.open(filename_archive, 'r:gz') as tar:
					if self._reload and self._verbose:
						print(filename + ' Downloaded.')
					subpath = os.path.join(
						path, tar.getnames()[0].split('/')[0]
					)
					if not osp.exists(subpath) or self._reload:
						tar.extractall(path=path)
					return subpath
		elif filename.endswith('.tar'):
			if tarfile.is_tarfile(filename_archive):
				with tarfile.open(filename_archive, 'r:') as tar:
					if self._reload and self._verbose:
						print(filename + ' Downloaded.')
					subpath = os.path.join(path, tar.getnames()[0])
					if not osp.exists(subpath) or self._reload:
						tar.extractall(path=path)
					return subpath
		elif filename.endswith('.zip'):
			with ZipFile(filename_archive, 'r') as zip_ref:
				if self._reload and self._verbose:
					print(filename + ' Downloaded.')
				subpath = os.path.join(path, zip_ref.namelist()[0])
				if not osp.exists(subpath) or self._reload:
					zip_ref.extractall(path)
				return subpath
		else:
			raise ValueError(filename + ' Unsupported file.')


	def get_all_ds_infos(self, database):
		"""Get information of all datasets from a database.

		Parameters
		----------
		database : string
			DESCRIPTION.

		Returns
		-------
		None.
		"""
		if database.lower() == 'tudataset':
			infos = self.get_all_tud_ds_infos()
		elif database.lower() == 'iam':
			pass
		else:
			msg = 'Invalid Database name "' + database + '"'
			msg += '\n Available databases are as follows: \n\n'
			msg += '\n'.join(db for db in sorted(DATABASES))
			msg += 'Check "gklearn.dataset.DATASET_META" for more details.'
			raise ValueError(msg)

		return infos


	def get_all_tud_ds_infos(self):
		"""Get information of all datasets from database TUDataset.

		Returns
		-------
		None.
		"""
		import urllib.request
		import urllib.error
		from lxml import etree

		try:
			response = urllib.request.urlopen(DATABASES['tudataset'])
		except urllib.error.HTTPError:
			print(
				'The URL of the database "TUDataset" is not available:\n' +
				DATABASES['tudataset']
				)

		infos = {}

		# Get tables.
		h_str = response.read()
		tree = etree.HTML(h_str)
		tables = tree.xpath('//table')
		for table in tables:
			# Get the domain of the datasets.
			h2_nodes = table.getprevious()
			if h2_nodes is not None and h2_nodes.tag == 'h2':
				domain = h2_nodes.text.strip().lower()
			else:
				domain = ''

			# Get each line in the table.
			tr_nodes = table.xpath('tbody/tr')
			for tr in tr_nodes[1:]:
				# Get each element in the line.
				td_node = tr.xpath('td')

				# task type.
				cls_txt = td_node[3].text.strip()
				if not cls_txt.startswith('R'):
					class_number = int(cls_txt)
					task_type = 'classification'
				else:
					class_number = None
					task_type = 'regression'

				# node attrs.
				na_text = td_node[8].text.strip()
				if not na_text.startswith('+'):
					node_attr_dim = 0
				else:
					node_attr_dim = int(re.findall('\((.*)\)', na_text)[0])

				# edge attrs.
				ea_text = td_node[10].text.strip()
				if ea_text == 'temporal':
					edge_attr_dim = ea_text
				elif not ea_text.startswith('+'):
					edge_attr_dim = 0
				else:
					edge_attr_dim = int(re.findall('\((.*)\)', ea_text)[0])

				# geometry.
				geo_txt = td_node[9].text.strip()
				if geo_txt == '–':
					geometry = None
				else:
					geometry = geo_txt

				# url.
				url = td_node[11].xpath('a')[0].attrib['href'].strip()
				pos_zip = url.rfind('.zip')
				url = url[:pos_zip + 4]

				infos[td_node[0].xpath('strong')[0].text.strip()] = {
					'database': 'tudataset',
					'reference': td_node[1].text.strip(),
					'dataset_size': int(td_node[2].text.strip()),
					'class_number': class_number,
					'task_type': task_type,
					'ave_node_num': float(td_node[4].text.strip()),
					'ave_edge_num': float(td_node[5].text.strip()),
					'node_labeled': True if td_node[
						                        6].text.strip() == '+' else False,
					'edge_labeled': True if td_node[
						                        7].text.strip() == '+' else False,
					'node_attr_dim': node_attr_dim,
					'geometry': geometry,
					'edge_attr_dim': edge_attr_dim,
					'url': url,
					'domain': domain
				}

		return infos


	def pretty_ds_infos(self, infos):
		"""Get the string that pretty prints the information of datasets.

		Parameters
		----------
		datasets : dict
			The datasets' information.

		Returns
		-------
		p_str : string
			The pretty print of the datasets' information.
		"""
		p_str = '{\n'
		for key, val in infos.items():
			p_str += '\t\'' + str(key) + '\': {\n'
			for k, v in val.items():
				p_str += '\t\t\'' + str(k) + '\': '
				if isinstance(v, str):
					p_str += '\'' + str(v) + '\',\n'
				else:
					p_str += '' + str(v) + ',\n'
			p_str += '\t},\n'
		p_str += '}'

		return p_str


	@property
	def path(self):
		return self._path


	def dataset(self):
		if self.mode == "Tensorflow":
			return  # something
		if self.mode == "Pytorch":
			return self.pytorch_dataset
		return self.dataset


	def info(self):
		print(self.info_dataset[self._name])


	def iter_load_dataset(self, data):
		results = []
		for datasets in data:
			results.append(
				loadDataset(osp.join(self._root, self._name, datasets))
			)
		return results


	def load_dataset(self, list_files):
		if self._name == "Ptc":
			if type(self.option) != str or self.option.upper() not in ['FR',
			                                                           'FM',
			                                                           'MM',
			                                                           'MR']:
				raise ValueError(
					'option for Ptc dataset needs to be one of : \n fr fm mm mr'
				)
			results = []
			results.append(
				loadDataset(
					osp.join(
						self.root, self._name, 'PTC/Test', self.gender + '.ds'
					                                       )
				)
			)
			results.append(
				loadDataset(
					osp.join(
						self.root, self._name, 'PTC/Train', self.gender + '.ds'
					                                        )
				)
			)
			return results
		if self.name == "Pah":
			maximum_sets = 0
			for file in list_files:
				if file.endswith('ds'):
					maximum_sets = max(
						maximum_sets, int(file.split('_')[1].split('.')[0])
					)
			self.max_for_letter = maximum_sets
			if not type(
					self.option
			) == int or self.option > maximum_sets or self.option < 0:
				raise ValueError(
					'option needs to be an integer between 0 and ' + str(
						maximum_sets
					)
					)
			data = self.has_train_valid_test["Pah"]
			data[0] = self.has_train_valid_test["Pah"][0].split('_')[
				          0] + '_' + str(self.option) + '.ds'
			data[1] = self.has_train_valid_test["Pah"][1].split('_')[
				          0] + '_' + str(self.option) + '.ds'
			return self.iter_load_dataset(data)
		if self.name == "Letter":
			if type(self.option) == str and self.option.upper() in \
					self.has_train_valid_test["Letter"]:
				data = self.has_train_valid_test["Letter"][self.option.upper()]
			else:
				message = "The parameter for letter is incorrect choose between : "
				message += "\nhigh  med  low"
				raise ValueError(message)
			return self.iter_load_dataset(data)
		if self.name in self.has_train_valid_test:  # common IAM dataset with train, valid and test
			data = self.has_train_valid_test[self.name]
			return self.iter_load_dataset(data)
		else:  # common dataset without train,valid and test, only dataset.ds file
			data = self.data_to_use_in_datasets[self.name]
			if len(data) > 1 and data[0] in list_files and data[
				1] in list_files:  # case for Alkane
				return loadDataset(
					osp.join(self.root, self.name, data[0]),
					filename_y=osp.join(self.root, self.name, data[1])
				)
			if data in list_files:
				return loadDataset(osp.join(self.root, self.name, data))


	def build_dictionary(self, Gs):
		labels = set()
		# next line : from DeepGraphWithNNTorch
		# bond_type_number_maxi = int(max(max([[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])))
		sizes = set()
		for G in Gs:
			for _, node in G.nodes(data=True):  # or for node in nx.nodes(G)
				# print(_,node)
				labels.add(
					node["label"][0]
				)  # labels.add(G.nodes[node]["label"][0])   #what do we use for IAM datasets (they don't have bond_type or event label) ?
			sizes.add(G.order())
		label_dict = {}
		# print("labels : ", labels, bond_type_number_maxi)
		for i, label in enumerate(labels):
			label_dict[label] = [0.] * len(labels)
			label_dict[label][i] = 1.
		return label_dict


	def from_networkx_to_pytorch(self, Gs, y):
		# exemple for MAO: atom_to_onehot = {'C': [1., 0., 0.], 'N': [0., 1., 0.], 'O': [0., 0., 1.]}
		# code from https://github.com/bgauzere/pygnn/blob/master/utils.py
		atom_to_onehot = self.build_dictionary(Gs)
		max_size = 30
		adjs = []
		inputs = []
		for i, G in enumerate(Gs):
			I = torch.eye(G.order(), G.order())
			# A = torch.Tensor(nx.adjacency_matrix(G).todense())
			# A = torch.Tensor(nx.to_numpy_matrix(G))
			A = torch.tensor(
				nx.to_scipy_sparse_matrix(
					G, dtype=int, weight='bond_type'
				).todense(), dtype=torch.int
			)  # what do we use for IAM datasets (they don't have bond_type or event label) ?
			adj = F.pad(
				A, pad=(0, max_size - G.order(), 0, max_size - G.order())
			)  # add I now ? if yes : F.pad(A + I,pad = (...))
			adjs.append(adj)

			f_0 = []
			for _, label in G.nodes(data=True):
				# print(_,label)
				cur_label = atom_to_onehot[label['label'][0]].copy()
				f_0.append(cur_label)

			X = F.pad(torch.Tensor(f_0), pad=(0, 0, 0, max_size - G.order()))
			inputs.append(X)
		return inputs, adjs, y


	def from_pytorch_to_tensorflow(self, batch_size):
		seed = random.randrange(sys.maxsize)
		random.seed(seed)
		tf_inputs = random.sample(self.pytorch_dataset[0], batch_size)
		random.seed(seed)
		tf_y = random.sample(self.pytorch_dataset[2], batch_size)


	def from_networkx_to_tensor(self, G, dict):
		A = nx.to_numpy_matrix(G)
		lab = [dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]
		return (
		torch.tensor(A).view(1, A.shape[0] * A.shape[1]), torch.tensor(lab))

# dataset= selfopen_files()
# print(build_dictionary(Gs))
# dic={'C':0,'N':1,'O':2}
# A,labels=from_networkx_to_tensor(Gs[13],dic)
# print(nx.to_numpy_matrix(Gs[13]),labels)
# print(A,labels)

# @todo : from_networkx_to_tensorflow


# dataloader = DataLoader('Acyclic',root = "database",option = 'high',mode = "Pytorch")
# dataloader.info()
# inputs,adjs,y = dataloader.pytorch_dataset

# """
# test,train,valid = dataloader.dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3,y3 = valid
# """
# #Gs,y = dataloader.
# #print(Gs,y)
# """
# Gs,y = dataloader.dataset
# for G in Gs :
# 	for e in G.edges():
# 		print(G[e[0]])

# """

# #for e in Gs[13].edges():
# #	print(Gs[13][e[0]])

# #print(from_networkx_to_tensor(Gs[7],{'C': [1., 0., 0.], 'N': [0., 1., 0.], 'O': [0., 0., 1.]}))

# #dataset.open_files()


# import os
# import os.path as osp
# import urllib
# import tarfile
# from zipfile import ZipFile
# from gklearn.utils.graphfiles import loadDataset
# import torch
# import torch.nn.functional as F
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np


#
# def DataLoader(name,root = 'data',mode = "Networkx",downloadAll = False,reload = False,letter = "High",number = 0,gender = "MM"):
# 	dir_name = "_".join(name.split("-"))
# 	if not osp.exists(root) :
# 		os.makedirs(root)
# 	url = "https://brunl01.users.greyc.fr/CHEMISTRY/"
# 	urliam = "https://iapr-tc15.greyc.fr/IAM/"
# 	list_database = {
# 		"Ace" : (url,"ACEDataset.tar"),
# 		"Acyclic" : (url,"Acyclic.tar.gz"),
# 		"Aids" : (urliam,"AIDS.zip"),
# 		"Alkane" : (url,"alkane_dataset.tar.gz"),
# 		"Chiral" : (url,"DatasetAcyclicChiral.tar"),
# 		"Coil_Del" : (urliam,"COIL-DEL.zip"),
# 		"Coil_Rag" : (urliam,"COIL-RAG.zip"),
# 		"Fingerprint" : (urliam,"Fingerprint.zip"),
# 		"Grec" : (urliam,"GREC.zip"),
# 		"Letter" : (urliam,"Letter.zip"),
# 		"Mao" : (url,"mao.tgz"),
# 		"Monoterpenoides" : (url,"monoterpenoides.tar.gz"),
# 		"Mutagenicity" : (urliam,"Mutagenicity.zip"),
# 		"Pah" : (url,"PAH.tar.gz"),
# 		"Protein" : (urliam,"Protein.zip"),
# 		"Ptc" : (url,"ptc.tgz"),
# 		"Steroid" : (url,"SteroidDataset.tar"),
# 		"Vitamin" : (url,"DatasetVitamin.tar"),
# 		"Web" : (urliam,"Web.zip")
# 	}
#
# 	data_to_use_in_datasets = {
# 		"Acyclic" : ("Acyclic/dataset_bps.ds"),
# 		"Aids" : ("AIDS_A.txt"),
# 		"Alkane" : ("Alkane/dataset.ds","Alkane/dataset_boiling_point_names.txt"),
# 		"Mao" : ("MAO/dataset.ds"),
# 		"Monoterpenoides" : ("monoterpenoides/dataset_10+.ds"), #('monoterpenoides/dataset.ds'),('monoterpenoides/dataset_9.ds'),('monoterpenoides/trainset_9.ds')
#
# 	}
# 	has_train_valid_test = {
# 		"Coil_Del" : ('COIL-DEL/data/test.cxl','COIL-DEL/data/train.cxl','COIL-DEL/data/valid.cxl'),
# 		"Coil_Rag" : ('COIL-RAG/data/test.cxl','COIL-RAG/data/train.cxl','COIL-RAG/data/valid.cxl'),
# 		"Fingerprint" : ('Fingerprint/data/test.cxl','Fingerprint/data/train.cxl','Fingerprint/data/valid.cxl'),
# 		"Grec" : ('GREC/data/test.cxl','GREC/data/train.cxl','GREC/data/valid.cxl'),
# 		"Letter" : {'HIGH' : ('Letter/HIGH/test.cxl','Letter/HIGH/train.cxl','Letter/HIGH/validation.cxl'),
# 					'MED' : ('Letter/MED/test.cxl','Letter/MED/train.cxl','Letter/MED/validation.cxl'),
# 					'LOW' : ('Letter/LOW/test.cxl','Letter/LOW/train.cxl','Letter/LOW/validation.cxl')
# 				   },
# 		"Mutagenicity" : ('Mutagenicity/data/test.cxl','Mutagenicity/data/train.cxl','Mutagenicity/data/validation.cxl'),
# 		"Pah" : ['PAH/testset_0.ds','PAH/trainset_0.ds'],
# 		"Protein" : ('Protein/data/test.cxl','Protein/data/train.cxl','Protein/data/valid.cxl'),
# 		"Web" : ('Web/data/test.cxl','Web/data/train.cxl','Web/data/valid.cxl')
# 	}
#
# 	if not name :
# 		raise ValueError("No dataset entered")
# 	if name not in list_database:
# 		message = "Invalid Dataset name " + name
# 		message += '\n Available datasets are as follows : \n\n'

# 		message += '\n'.join(database for database in list_database)
# 		raise ValueError(message)
#
# 	def download_file(url,filename):
# 		try :
# 			response = urllib.request.urlopen(url + filename)
# 		except urllib.error.HTTPError:
# 			print(filename + " not available or incorrect http link")
# 			return
# 		return response
#
# 	def write_archive_file(root,database):
# 		path = osp.join(root,database)
# 		url,filename = list_database[database]
# 		filename_dir = osp.join(path,filename)
# 		if not osp.exists(filename_dir) or reload:
# 			response = download_file(url,filename)
# 			if response is None :
# 				return
# 			if not osp.exists(path) :
# 				os.makedirs(path)
# 			with open(filename_dir,'wb') as outfile :
# 				outfile.write(response.read())
#
# 	if downloadAll :
# 		print('Waiting...')
# 		for database in list_database :
# 			write_archive_file(root,database)
# 		print('Downloading finished')
# 	else:
# 		write_archive_file(root,name)
#
# 	def iter_load_dataset(data):
# 		results = []
# 		for datasets in data :
# 			results.append(loadDataset(osp.join(root,name,datasets)))
# 		return results
#
# 	def load_dataset(list_files):
# 		if name == "Ptc":
# 			if gender.upper() not in ['FR','FM','MM','MR']:
# 				raise ValueError('gender chosen needs to be one of \n fr fm mm mr')
# 			results = []
# 			results.append(loadDataset(osp.join(root,name,'PTC/Test',gender.upper() + '.ds')))
# 			results.append(loadDataset(osp.join(root,name,'PTC/Train',gender.upper() + '.ds')))
# 			return results
# 		if name == "Pah":
# 			maximum_sets = 0
# 			for file in list_files:
# 				if file.endswith('ds'):
# 					maximum_sets = max(maximum_sets,int(file.split('_')[1].split('.')[0]))
# 			if number > maximum_sets :
# 				raise ValueError("Please select a dataset with number less than " + str(maximum_sets + 1))
# 			data = has_train_valid_test["Pah"]
# 			data[0] = has_train_valid_test["Pah"][0].split('_')[0] + '_' + str(number) + '.ds'
# 			data[1] = has_train_valid_test["Pah"][1].split('_')[0] + '_' + str(number) + '.ds'
# 			#print(data)
# 			return iter_load_dataset(data)
# 		if name == "Letter":
# 			if letter.upper() in has_train_valid_test["Letter"]:
# 				data = has_train_valid_test["Letter"][letter.upper()]
# 			else:
# 				message = "The parameter for letter is incorrect choose between : "
# 				message += "\nhigh  med  low"
# 				raise ValueError(message)
# 			results = []
# 			for datasets in data:
# 				results.append(loadDataset(osp.join(root,name,datasets)))
# 			return results
# 		if name in has_train_valid_test : #common IAM dataset with train, valid and test
# 			data = has_train_valid_test[name]
# 			results = []
# 			for datasets in data :
# 				results.append(loadDataset(osp.join(root,name,datasets)))
# 			return results
# 		else:  #common dataset without train,valid and test, only dataset.ds file
# 			data = data_to_use_in_datasets[name]
# 			if len(data) > 1 and data[0] in list_files and data[1] in list_files:
# 				return loadDataset(osp.join(root,name,data[0]),filename_y = osp.join(root,name,data[1]))
# 			if data in list_files:
# 				return loadDataset(osp.join(root,name,data))

# 	def open_files():
# 		filename = list_database[name][1]
# 		path = osp.join(root,name)
# 		filename_archive = osp.join(root,name,filename)
#
# 		if filename.endswith('gz'):
# 			if tarfile.is_tarfile(filename_archive):
# 				with tarfile.open(filename_archive,"r:gz") as tar:
# 					if reload:
# 						print(filename + " Downloaded")
# 					tar.extractall(path = path)
# 					return load_dataset(tar.getnames())
# 					#raise ValueError("dataset not available")
#
#
# 		elif filename.endswith('.tar'):
# 			if tarfile.is_tarfile(filename_archive):
# 				with tarfile.open(filename_archive,"r:") as tar:
# 					if reload :
# 						print(filename + " Downloaded")
# 					tar.extractall(path = path)
# 					return load_dataset(tar.getnames())
# 		elif filename.endswith('.zip'):
# 			with ZipFile(filename_archive,"r") as zip_ref:
# 				if reload :
# 						print(filename + " Downloaded")
# 				zip_ref.extractall(path)
# 				return load_dataset(zip_ref.namelist())
# 		else:
# 			print(filename + " Unsupported file")
# 		"""
# 		with tarfile.open(osp.join(root,name,list_database[name][1]),"r:gz") as files:
# 			for file in files.getnames():
# 				print(file)
# 		"""
#
# 	def build_dictionary(Gs):
# 		labels = set()
# 		bond_type_number_maxi = int(max(max([[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])))
# 		print(bond_type_number_maxi)
# 		sizes = set()
# 		for G in Gs :
# 			for _,node in G.nodes(data = True): # or for node in nx.nodes(G)
# 				#print(node)
# 				labels.add(node["label"][0]) # labels.add(G.nodes[node]["label"][0])
# 			sizes.add(G.order())
# 			if len(labels) >= bond_type_number_maxi:
# 				break
# 		label_dict = {}
# 		for i,label in enumerate(labels):
# 			label_dict[label] = [0.]*bond_type_number_maxi
# 			label_dict[label][i] = 1.
# 		return label_dict
#
# 	def from_networkx_to_pytorch(Gs):
# 		#exemple : atom_to_onehot = {'C': [1., 0., 0.], 'N': [0., 1., 0.], 'O': [0., 0., 1.]}
# 		# code from https://github.com/bgauzere/pygnn/blob/master/utils.py
# 		atom_to_onehot = build_dictionary(Gs)
# 		max_size = 30
# 		adjs = []
# 		inputs = []
# 		for i, G in enumerate(Gs):
# 			I = torch.eye(G.order(), G.order())
# 			A = torch.Tensor(nx.adjacency_matrix(G).todense())
# 			A = torch.tensor(nx.to_scipy_sparse_matrix(G,dtype = int,weight = 'bond_type').todense(),dtype = torch.int)
# 			adj = F.pad(A+I, pad=(0, max_size-G.order(), 0, max_size-G.order()))  #add I now ?
# 			adjs.append(adj)

# 			f_0 = []
# 			for _, label in G.nodes(data=True):
# 				#print(_,label)
# 				cur_label = atom_to_onehot[label['label'][0]].copy()
# 				f_0.append(cur_label)

# 			X = F.pad(torch.Tensor(f_0), pad=(0, 0, 0, max_size-G.order()))
# 			inputs.append(X)
# 		return inputs,adjs,y
#
# 	def from_networkx_to_tensor(G,dict):

# 		A=nx.to_numpy_matrix(G)
# 		lab=[dict[G.nodes[v]['label'][0]] for v in nx.nodes(G)]
# 		return (torch.tensor(A).view(1,A.shape[0]*A.shape[1]),torch.tensor(lab))
#
# 	dataset= open_files()
# 	#print(build_dictionary(Gs))
# 	#dic={'C':0,'N':1,'O':2}
# 	#A,labels=from_networkx_to_tensor(Gs[13],dic)
# 	#print(nx.to_numpy_matrix(Gs[13]),labels)
# 	#print(A,labels)
#
# 	"""
# 	for G in Gs :
# 		for node in nx.nodes(G):
# 			print(G.nodes[node])
# 	"""
# 	if mode == "pytorch":
# 		Gs,y = dataset
# 		inputs,adjs,y = from_networkx_to_pytorch(Gs)
# 		print(inputs,adjs)
# 		return inputs,adjs,y
#
#
# 		"""
# 		dic = dict()
# 		for i,l in enumerate(label):
# 			dic[l] = i
# 		dic = {'C': 0, 'N': 1, 'O': 2}
# 		A,labels=from_networkx_to_tensor(Gs[0],dic)
# 		#print(A,labels)
# 		return A,labels
# 		"""
#
# 	return dataset
#
# 	#open_files()
#

# def label_to_color(label):
# 	if label == 'C':
# 		return 0.1
# 	elif label == 'O':
# 		return 0.8
#
# def nodes_to_color_sequence(G):
# 	return [label_to_color(c[1]['label'][0]) for c in G.nodes(data=True)]


# ##############
# """
# dataset = DataLoader('Mao',root = "database")
# print(dataset)
# Gs,y = dataset
# """

# """
# dataset = DataLoader('Alkane',root = "database") # Gs is empty here whereas y isn't -> not working
# Gs,y = dataset
# """

# """
# dataset = DataLoader('Acyclic', root = "database")
# Gs,y = dataset
# """

# """
# dataset = DataLoader('Monoterpenoides', root = "database")
# Gs,y = dataset
# """

# """
# dataset = DataLoader('Pah',root = 'database', number = 8)
# test_set,train_set = dataset
# Gs,y = test_set
# Gs2,y2 = train_set
# """

# """
# dataset = DataLoader('Coil_Del',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """

# """
# dataset = DataLoader('Coil_Rag',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """

# """
# dataset = DataLoader('Fingerprint',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """

# """
# dataset = DataLoader('Grec',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """

# """
# dataset = DataLoader('Letter',root = "database",letter = 'low') #high low med
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """

# """
# dataset = DataLoader('Mutagenicity',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """
# """
# dataset = DataLoader('Protein',root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3, y3 = valid
# """


# """
# dataset = DataLoader('Ptc', root = "database",gender = 'fm')  # not working, Gs and y are empty perhaps issue coming from loadDataset
# valid,train = dataset
# Gs,y = valid
# Gs2,y2 = train
# """

# """
# dataset = DataLoader('Web', root = "database")
# test,train,valid = dataset
# Gs,y = test
# Gs2,y2 = train
# Gs3,y3 = valid
# """
# print(Gs,y)
# print(len(dataset))
# ##############
# #print('edge max label',max(max([[G[e[0]][e[1]]['bond_type'] for e in G.edges()] for G in Gs])))
# G1 = Gs[13]
# G2 = Gs[23]
# """
# nx.draw_networkx(G1,with_labels=True,node_color = nodes_to_color_sequence(G1),cmap='autumn')
# plt.figure()

# nx.draw_networkx(G2,with_labels=True,node_color = nodes_to_color_sequence(G2),cmap='autumn')
# """


# from pathlib import Path

# DATA_PATH = Path("data")

# def import_datasets():
#
# 	import urllib
# 	import tarfile
# 	from zipfile import ZipFile

# 	URL = "https://brunl01.users.greyc.fr/CHEMISTRY/"
# 	URLIAM = "https://iapr-tc15.greyc.fr/IAM/"
#

# 	LIST_DATABASE = {
# 		"Pah" : (URL,"PAH.tar.gz"),
# 		"Mao" : (URL,"mao.tgz"),
# 		"Ptc" : (URL,"ptc.tgz"),
# 		"Aids" : (URLIAM,"AIDS.zip"),
# 		"Acyclic" : (URL,"Acyclic.tar.gz"),
# 		"Alkane" : (URL,"alkane_dataset.tar.gz"),
# 		"Chiral" : (URL,"DatasetAcyclicChiral.tar"),
# 		"Vitamin" : (URL,"DatasetVitamin.tar"),
# 		"Ace" : (URL,"ACEDataset.tar"),
# 		"Steroid" : (URL,"SteroidDataset.tar"),
# 		"Monoterpenoides" : (URL,"monoterpenoides.tar.gz"),
# 		"Letter" : (URLIAM,"Letter.zip"),
# 		"Grec" : (URLIAM,"GREC.zip"),
# 		"Fingerprint" : (URLIAM,"Fingerprint.zip"),
# 		"Coil_Rag" : (URLIAM,"COIL-RAG.zip"),
# 		"Coil_Del" : (URLIAM,"COIL-DEL.zip"),
# 		"Web" : (URLIAM,"Web.zip"),
# 		"Mutagenicity" : (URLIAM,"Mutagenicity.zip"),
# 		"Protein" : (URLIAM,"Protein.zip")
# 	}
# 	print("Select databases in the list. Select multiple, split by white spaces .\nWrite All to select all of them.\n")
# 	print(', '.join(database for database in LIST_DATABASE))

# 	print("Choice : ",end = ' ')
# 	selected_databases = input().split()

#
# 	def download_file(url,filename):
# 		try :
# 			response = urllib.request.urlopen(url + filename)
# 		except urllib.error.HTTPError:
# 			print(filename + " not available or incorrect http link")
# 			return
# 		return response
#
# 	def write_archive_file(database):
#
# 		PATH = DATA_PATH / database
# 		url,filename = LIST_DATABASE[database]
# 		if not (PATH / filename).exists():
# 			response = download_file(url,filename)
# 			if response is None :
# 				return
# 			if not PATH.exists() :
# 				PATH.mkdir(parents=True, exist_ok=True)
# 			with open(PATH/filename,'wb') as outfile :
# 				outfile.write(response.read())
#
# 			if filename[-2:] == 'gz':
# 				if tarfile.is_tarfile(PATH/filename):
# 					with tarfile.open(PATH/filename,"r:gz") as tar:
# 						tar.extractall(path = PATH)
# 						print(filename + '   Downloaded')
# 			elif filename[-3:] == 'tar':
# 				if tarfile.is_tarfile(PATH/filename):
# 					with tarfile.open(PATH/filename,"r:") as tar:
# 						tar.extractall(path = PATH)
# 						print(filename + '   Downloaded')
# 			elif filename[-3:] == 'zip':
# 				with ZipFile(PATH/filename,"r") as zip_ref:
# 					zip_ref.extractall(PATH)
# 					print(filename + '   Downloaded')
# 			else:
# 				print("Unsupported file")

# 	if 'All' in selected_databases:
# 		print('Waiting...')
# 		for database in LIST_DATABASE :
# 			write_archive_file(database)
# 		print('Finished')
# 	else:
# 		print('Waiting...')
# 		for database in selected_databases :
# 			if database in LIST_DATABASE :
# 				write_archive_file(database)
# 		print('Finished')
# import_datasets()


# class GraphFetcher(object):
#
#
# 	def __init__(self, filename=None, filename_targets=None, **kwargs):
# 		if filename is None:
# 			self._graphs = None
# 			self._targets = None
# 			self._node_labels = None
# 			self._edge_labels = None
# 			self._node_attrs = None
# 			self._edge_attrs = None
# 		else:
# 			self.load_dataset(filename, filename_targets=filename_targets, **kwargs)
#
# 		self._substructures = None
# 		self._node_label_dim = None
# 		self._edge_label_dim = None
# 		self._directed = None
# 		self._dataset_size = None
# 		self._total_node_num = None
# 		self._ave_node_num = None
# 		self._min_node_num = None
# 		self._max_node_num = None
# 		self._total_edge_num = None
# 		self._ave_edge_num = None
# 		self._min_edge_num = None
# 		self._max_edge_num = None
# 		self._ave_node_degree = None
# 		self._min_node_degree = None
# 		self._max_node_degree = None
# 		self._ave_fill_factor = None
# 		self._min_fill_factor = None
# 		self._max_fill_factor = None
# 		self._node_label_nums = None
# 		self._edge_label_nums = None
# 		self._node_attr_dim = None
# 		self._edge_attr_dim = None
# 		self._class_number = None
#
#
# 	def load_dataset(self, filename, filename_targets=None, **kwargs):
# 		self._graphs, self._targets, label_names = load_dataset(filename, filename_targets=filename_targets, **kwargs)
# 		self._node_labels = label_names['node_labels']
# 		self._node_attrs = label_names['node_attrs']
# 		self._edge_labels = label_names['edge_labels']
# 		self._edge_attrs = label_names['edge_attrs']
# 		self.clean_labels()
#
#
# 	def load_graphs(self, graphs, targets=None):
# 		# this has to be followed by set_labels().
# 		self._graphs = graphs
# 		self._targets = targets
# #		self.set_labels_attrs() # @todo
#
#
# 	def load_predefined_dataset(self, ds_name):
# 		current_path = os.path.dirname(os.path.realpath(__file__)) + '/'
# 		if ds_name == 'Acyclic':
# 			ds_file = current_path + '../../datasets/Acyclic/dataset_bps.ds'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'AIDS':
# 			ds_file = current_path + '../../datasets/AIDS/AIDS_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Alkane':
# 			ds_file = current_path + '../../datasets/Alkane/dataset.ds'
# 			fn_targets = current_path + '../../datasets/Alkane/dataset_boiling_point_names.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file, filename_targets=fn_targets)
# 		elif ds_name == 'COIL-DEL':
# 			ds_file = current_path + '../../datasets/COIL-DEL/COIL-DEL_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'COIL-RAG':
# 			ds_file = current_path + '../../datasets/COIL-RAG/COIL-RAG_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'COLORS-3':
# 			ds_file = current_path + '../../datasets/COLORS-3/COLORS-3_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Cuneiform':
# 			ds_file = current_path + '../../datasets/Cuneiform/Cuneiform_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'DD':
# 			ds_file = current_path + '../../datasets/DD/DD_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'ENZYMES':
# 			ds_file = current_path + '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Fingerprint':
# 			ds_file = current_path + '../../datasets/Fingerprint/Fingerprint_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'FRANKENSTEIN':
# 			ds_file = current_path + '../../datasets/FRANKENSTEIN/FRANKENSTEIN_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Letter-high': # node non-symb
# 			ds_file = current_path + '../../datasets/Letter-high/Letter-high_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Letter-low': # node non-symb
# 			ds_file = current_path + '../../datasets/Letter-low/Letter-low_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Letter-med': # node non-symb
# 			ds_file = current_path + '../../datasets/Letter-med/Letter-med_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'MAO':
# 			ds_file = current_path + '../../datasets/MAO/dataset.ds'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Monoterpenoides':
# 			ds_file = current_path + '../../datasets/Monoterpenoides/dataset_10+.ds'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'MUTAG':
# 			ds_file = current_path + '../../datasets/MUTAG/MUTAG_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'NCI1':
# 			ds_file = current_path + '../../datasets/NCI1/NCI1_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'NCI109':
# 			ds_file = current_path + '../../datasets/NCI109/NCI109_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'PAH':
# 			ds_file = current_path + '../../datasets/PAH/dataset.ds'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'SYNTHETIC':
# 			pass
# 		elif ds_name == 'SYNTHETICnew':
# 			ds_file = current_path + '../../datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
# 			self._graphs, self._targets, label_names = load_dataset(ds_file)
# 		elif ds_name == 'Synthie':
# 			pass
# 		else:
# 			raise Exception('The dataset name "', ds_name, '" is not pre-defined.')
#
# 		self._node_labels = label_names['node_labels']
# 		self._node_attrs = label_names['node_attrs']
# 		self._edge_labels = label_names['edge_labels']
# 		self._edge_attrs = label_names['edge_attrs']
# 		self.clean_labels()
#

# 	def set_labels(self, node_labels=[], node_attrs=[], edge_labels=[], edge_attrs=[]):
# 		self._node_labels = node_labels
# 		self._node_attrs = node_attrs
# 		self._edge_labels = edge_labels
# 		self._edge_attrs = edge_attrs

#
# 	def set_labels_attrs(self, node_labels=None, node_attrs=None, edge_labels=None, edge_attrs=None):
# 		# @todo: remove labels which have only one possible values.
# 		if node_labels is None:
# 			self._node_labels = self._graphs[0].graph['node_labels']
# #			# graphs are considered node unlabeled if all nodes have the same label.
# #			infos.update({'node_labeled': is_nl if node_label_num > 1 else False})
# 		if node_attrs is None:
# 			self._node_attrs = self._graphs[0].graph['node_attrs']
# #		for G in Gn:
# #			for n in G.nodes(data=True):
# #				if 'attributes' in n[1]:
# #					return len(n[1]['attributes'])
# #		return 0
# 		if edge_labels is None:
# 			self._edge_labels = self._graphs[0].graph['edge_labels']
# #			# graphs are considered edge unlabeled if all edges have the same label.
# #			infos.update({'edge_labeled': is_el if edge_label_num > 1 else False})
# 		if edge_attrs is None:
# 			self._edge_attrs = self._graphs[0].graph['edge_attrs']
# #		for G in Gn:
# #			if nx.number_of_edges(G) > 0:
# #				for e in G.edges(data=True):
# #					if 'attributes' in e[2]:
# #						return len(e[2]['attributes'])
# #		return 0
#
#
# 	def get_dataset_infos(self, keys=None, params=None):
# 		"""Computes and returns the structure and property information of the graph dataset.
#
# 		Parameters
# 		----------
# 		keys : list, optional
# 			A list of strings which indicate which informations will be returned. The
# 			possible choices includes:
#
# 			'substructures': sub-structures graphs contains, including 'linear', 'non
# 		linear' and 'cyclic'.
#
# 			'node_label_dim': whether vertices have symbolic labels.
#
# 			'edge_label_dim': whether egdes have symbolic labels.
#
# 			'directed': whether graphs in dataset are directed.
#
# 			'dataset_size': number of graphs in dataset.
#
# 			'total_node_num': total number of vertices of all graphs in dataset.
#
# 			'ave_node_num': average number of vertices of graphs in dataset.
#
# 			'min_node_num': minimum number of vertices of graphs in dataset.
#
# 			'max_node_num': maximum number of vertices of graphs in dataset.
#
# 			'total_edge_num': total number of edges of all graphs in dataset.
#
# 			'ave_edge_num': average number of edges of graphs in dataset.
#
# 			'min_edge_num': minimum number of edges of graphs in dataset.
#
# 			'max_edge_num': maximum number of edges of graphs in dataset.
#
# 			'ave_node_degree': average vertex degree of graphs in dataset.
#
# 			'min_node_degree': minimum vertex degree of graphs in dataset.
#
# 			'max_node_degree': maximum vertex degree of graphs in dataset.
#
# 			'ave_fill_factor': average fill factor (number_of_edges /
# 		(number_of_nodes ** 2)) of graphs in dataset.
#
# 			'min_fill_factor': minimum fill factor of graphs in dataset.
#
# 			'max_fill_factor': maximum fill factor of graphs in dataset.
#
# 			'node_label_nums': list of numbers of symbolic vertex labels of graphs in dataset.
#
# 			'edge_label_nums': list number of symbolic edge labels of graphs in dataset.
#
# 			'node_attr_dim': number of dimensions of non-symbolic vertex labels.
# 		Extracted from the 'attributes' attribute of graph nodes.
#
# 			'edge_attr_dim': number of dimensions of non-symbolic edge labels.
# 		Extracted from the 'attributes' attribute of graph edges.
#
# 			'class_number': number of classes. Only available for classification problems.
#
# 			'all_degree_entropy': the entropy of degree distribution of each graph.
#
# 			'ave_degree_entropy': the average entropy of degree distribution of all graphs.
#
# 			All informations above will be returned if `keys` is not given.
#
# 		params: dict of dict, optional
# 			A dictinary which contains extra parameters for each possible
# 			element in ``keys``.
#
# 		Return
# 		------
# 		dict
# 			Information of the graph dataset keyed by `keys`.
# 		"""
# 		infos = {}
#
# 		if keys == None:
# 			keys = [
# 				'substructures',
# 				'node_label_dim',
# 				'edge_label_dim',
# 				'directed',
# 				'dataset_size',
# 				'total_node_num',
# 				'ave_node_num',
# 				'min_node_num',
# 				'max_node_num',
# 				'total_edge_num',
# 				'ave_edge_num',
# 				'min_edge_num',
# 				'max_edge_num',
# 				'ave_node_degree',
# 				'min_node_degree',
# 				'max_node_degree',
# 				'ave_fill_factor',
# 				'min_fill_factor',
# 				'max_fill_factor',
# 				'node_label_nums',
# 				'edge_label_nums',
# 				'node_attr_dim',
# 				'edge_attr_dim',
# 				'class_number',
# 				'all_degree_entropy',
# 				'ave_degree_entropy'
# 			]
#
# 		# dataset size
# 		if 'dataset_size' in keys:
# 			if self._dataset_size is None:
# 				self._dataset_size = self._get_dataset_size()
# 			infos['dataset_size'] = self._dataset_size
#
# 		# graph node number
# 		if any(i in keys for i in ['total_node_num', 'ave_node_num', 'min_node_num', 'max_node_num']):
# 			all_node_nums = self._get_all_node_nums()

# 		if 'total_node_num' in keys:
# 			if self._total_node_num is None:
# 				self._total_node_num = self._get_total_node_num(all_node_nums)
# 			infos['total_node_num'] = self._total_node_num
#
# 		if 'ave_node_num' in keys:
# 			if self._ave_node_num is None:
# 				self._ave_node_num = self._get_ave_node_num(all_node_nums)
# 			infos['ave_node_num'] = self._ave_node_num
#
# 		if 'min_node_num' in keys:
# 			if self._min_node_num is None:
# 				self._min_node_num = self._get_min_node_num(all_node_nums)
# 			infos['min_node_num'] = self._min_node_num
#
# 		if 'max_node_num' in keys:
# 			if self._max_node_num is None:
# 				self._max_node_num = self._get_max_node_num(all_node_nums)
# 			infos['max_node_num'] = self._max_node_num
#
# 		# graph edge number
# 		if any(i in keys for i in ['total_edge_num', 'ave_edge_num', 'min_edge_num', 'max_edge_num']):
# 			all_edge_nums = self._get_all_edge_nums()

# 		if 'total_edge_num' in keys:
# 			if self._total_edge_num is None:
# 				self._total_edge_num = self._get_total_edge_num(all_edge_nums)
# 			infos['total_edge_num'] = self._total_edge_num
#
# 		if 'ave_edge_num' in keys:
# 			if self._ave_edge_num is None:
# 				self._ave_edge_num = self._get_ave_edge_num(all_edge_nums)
# 			infos['ave_edge_num'] = self._ave_edge_num
#
# 		if 'max_edge_num' in keys:
# 			if self._max_edge_num is None:
# 				self._max_edge_num = self._get_max_edge_num(all_edge_nums)
# 			infos['max_edge_num'] = self._max_edge_num

# 		if 'min_edge_num' in keys:
# 			if self._min_edge_num is None:
# 				self._min_edge_num = self._get_min_edge_num(all_edge_nums)
# 			infos['min_edge_num'] = self._min_edge_num
#
# 		# label number
# 		if 'node_label_dim' in keys:
# 			if self._node_label_dim is None:
# 				self._node_label_dim = self._get_node_label_dim()
# 			infos['node_label_dim'] = self._node_label_dim
#
# 		if 'node_label_nums' in keys:
# 			if self._node_label_nums is None:
# 				self._node_label_nums = {}
# 				for node_label in self._node_labels:
# 					self._node_label_nums[node_label] = self._get_node_label_num(node_label)
# 			infos['node_label_nums'] = self._node_label_nums
#
# 		if 'edge_label_dim' in keys:
# 			if self._edge_label_dim is None:
# 				self._edge_label_dim = self._get_edge_label_dim()
# 			infos['edge_label_dim'] = self._edge_label_dim
#
# 		if 'edge_label_nums' in keys:
# 			if self._edge_label_nums is None:
# 				self._edge_label_nums = {}
# 				for edge_label in self._edge_labels:
# 					self._edge_label_nums[edge_label] = self._get_edge_label_num(edge_label)
# 			infos['edge_label_nums'] = self._edge_label_nums
#
# 		if 'directed' in keys or 'substructures' in keys:
# 			if self._directed is None:
# 				self._directed = self._is_directed()
# 			infos['directed'] = self._directed
#
# 		# node degree
# 		if any(i in keys for i in ['ave_node_degree', 'max_node_degree', 'min_node_degree']):
# 			all_node_degrees = self._get_all_node_degrees()
#
# 		if 'ave_node_degree' in keys:
# 			if self._ave_node_degree is None:
# 				self._ave_node_degree = self._get_ave_node_degree(all_node_degrees)
# 			infos['ave_node_degree'] = self._ave_node_degree
#
# 		if 'max_node_degree' in keys:
# 			if self._max_node_degree is None:
# 				self._max_node_degree = self._get_max_node_degree(all_node_degrees)
# 			infos['max_node_degree'] = self._max_node_degree
#
# 		if 'min_node_degree' in keys:
# 			if self._min_node_degree is None:
# 				self._min_node_degree = self._get_min_node_degree(all_node_degrees)
# 			infos['min_node_degree'] = self._min_node_degree
#
# 		# fill factor
# 		if any(i in keys for i in ['ave_fill_factor', 'max_fill_factor', 'min_fill_factor']):
# 			all_fill_factors = self._get_all_fill_factors()
#
# 		if 'ave_fill_factor' in keys:
# 			if self._ave_fill_factor is None:
# 				self._ave_fill_factor = self._get_ave_fill_factor(all_fill_factors)
# 			infos['ave_fill_factor'] = self._ave_fill_factor
#
# 		if 'max_fill_factor' in keys:
# 			if self._max_fill_factor is None:
# 				self._max_fill_factor = self._get_max_fill_factor(all_fill_factors)
# 			infos['max_fill_factor'] = self._max_fill_factor
#
# 		if 'min_fill_factor' in keys:
# 			if self._min_fill_factor is None:
# 				self._min_fill_factor = self._get_min_fill_factor(all_fill_factors)
# 			infos['min_fill_factor'] = self._min_fill_factor
#
# 		if 'substructures' in keys:
# 			if self._substructures is None:
# 				self._substructures = self._get_substructures()
# 			infos['substructures'] = self._substructures
#
# 		if 'class_number' in keys:
# 			if self._class_number is None:
# 				self._class_number = self._get_class_number()
# 			infos['class_number'] = self._class_number
#
# 		if 'node_attr_dim' in keys:
# 			if self._node_attr_dim is None:
# 				self._node_attr_dim = self._get_node_attr_dim()
# 			infos['node_attr_dim'] = self._node_attr_dim
#
# 		if 'edge_attr_dim' in keys:
# 			if self._edge_attr_dim is None:
# 				self._edge_attr_dim = self._get_edge_attr_dim()
# 			infos['edge_attr_dim'] = self._edge_attr_dim
#
# 		# entropy of degree distribution.
#
# 		if 'all_degree_entropy' in keys:
# 			if params is not None and ('all_degree_entropy' in params) and ('base' in params['all_degree_entropy']):
# 				base = params['all_degree_entropy']['base']
# 			else:
# 				base = None
# 			infos['all_degree_entropy'] = self._compute_all_degree_entropy(base=base)
#
# 		if 'ave_degree_entropy' in keys:
# 			if params is not None and ('ave_degree_entropy' in params) and ('base' in params['ave_degree_entropy']):
# 				base = params['ave_degree_entropy']['base']
# 			else:
# 				base = None
# 			infos['ave_degree_entropy'] = np.mean(self._compute_all_degree_entropy(base=base))
#
# 		return infos
#
#
# 	def print_graph_infos(self, infos):
# 		from collections import OrderedDict
# 		keys = list(infos.keys())
# 		print(OrderedDict(sorted(infos.items(), key=lambda i: keys.index(i[0]))))
#
#
# 	def remove_labels(self, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
# 		node_labels = [item for item in node_labels if item in self._node_labels]
# 		edge_labels = [item for item in edge_labels if item in self._edge_labels]
# 		node_attrs = [item for item in node_attrs if item in self._node_attrs]
# 		edge_attrs = [item for item in edge_attrs if item in self._edge_attrs]

# 		for g in self._graphs:
# 			for nd in g.nodes():
# 				for nl in node_labels:
# 					del g.nodes[nd][nl]
# 				for na in node_attrs:
# 					del g.nodes[nd][na]
# 			for ed in g.edges():
# 				for el in edge_labels:
# 					del g.edges[ed][el]
# 				for ea in edge_attrs:
# 					del g.edges[ed][ea]
# 		if len(node_labels) > 0:
# 			self._node_labels = [nl for nl in self._node_labels if nl not in node_labels]
# 		if len(edge_labels) > 0:
# 			self._edge_labels = [el for el in self._edge_labels if el not in edge_labels]
# 		if len(node_attrs) > 0:
# 			self._node_attrs = [na for na in self._node_attrs if na not in node_attrs]
# 		if len(edge_attrs) > 0:
# 			self._edge_attrs = [ea for ea in self._edge_attrs if ea not in edge_attrs]
#
#
# 	def clean_labels(self):
# 		labels = []
# 		for name in self._node_labels:
# 			label = set()
# 			for G in self._graphs:
# 				label = label | set(nx.get_node_attributes(G, name).values())
# 				if len(label) > 1:
# 					labels.append(name)
# 					break
# 			if len(label) < 2:
# 				for G in self._graphs:
# 					for nd in G.nodes():
# 						del G.nodes[nd][name]
# 		self._node_labels = labels

# 		labels = []
# 		for name in self._edge_labels:
# 			label = set()
# 			for G in self._graphs:
# 				label = label | set(nx.get_edge_attributes(G, name).values())
# 				if len(label) > 1:
# 					labels.append(name)
# 					break
# 			if len(label) < 2:
# 				for G in self._graphs:
# 					for ed in G.edges():
# 						del G.edges[ed][name]
# 		self._edge_labels = labels

# 		labels = []
# 		for name in self._node_attrs:
# 			label = set()
# 			for G in self._graphs:
# 				label = label | set(nx.get_node_attributes(G, name).values())
# 				if len(label) > 1:
# 					labels.append(name)
# 					break
# 			if len(label) < 2:
# 				for G in self._graphs:
# 					for nd in G.nodes():
# 						del G.nodes[nd][name]
# 		self._node_attrs = labels

# 		labels = []
# 		for name in self._edge_attrs:
# 			label = set()
# 			for G in self._graphs:
# 				label = label | set(nx.get_edge_attributes(G, name).values())
# 				if len(label) > 1:
# 					labels.append(name)
# 					break
# 			if len(label) < 2:
# 				for G in self._graphs:
# 					for ed in G.edges():
# 						del G.edges[ed][name]
# 		self._edge_attrs = labels
#
#
# 	def cut_graphs(self, range_):
# 		self._graphs = [self._graphs[i] for i in range_]
# 		if self._targets is not None:
# 			self._targets = [self._targets[i] for i in range_]
# 		self.clean_labels()


# 	def trim_dataset(self, edge_required=False):
# 		if edge_required:
# 			trimed_pairs = [(idx, g) for idx, g in enumerate(self._graphs) if (nx.number_of_nodes(g) != 0 and nx.number_of_edges(g) != 0)]
# 		else:
# 			trimed_pairs = [(idx, g) for idx, g in enumerate(self._graphs) if nx.number_of_nodes(g) != 0]
# 		idx = [p[0] for p in trimed_pairs]
# 		self._graphs = [p[1] for p in trimed_pairs]
# 		self._targets = [self._targets[i] for i in idx]
# 		self.clean_labels()
#
#
# 	def copy(self):
# 		dataset = Dataset()
# 		graphs = [g.copy() for g in self._graphs] if self._graphs is not None else None
# 		target = self._targets.copy() if self._targets is not None else None
# 		node_labels = self._node_labels.copy() if self._node_labels is not None else None
# 		node_attrs = self._node_attrs.copy() if self._node_attrs is not None else None
# 		edge_labels = self._edge_labels.copy() if self._edge_labels is not None else None
# 		edge_attrs = self._edge_attrs.copy() if self._edge_attrs is not None else None
# 		dataset.load_graphs(graphs, target)
# 		dataset.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
# 		# @todo: clean_labels and add other class members?
# 		return dataset
#
#
# 	def get_all_node_labels(self):
# 		node_labels = []
# 		for g in self._graphs:
# 			for n in g.nodes():
# 				nl = tuple(g.nodes[n].items())
# 				if nl not in node_labels:
# 					node_labels.append(nl)
# 		return node_labels
#
#
# 	def get_all_edge_labels(self):
# 		edge_labels = []
# 		for g in self._graphs:
# 			for e in g.edges():
# 				el = tuple(g.edges[e].items())
# 				if el not in edge_labels:
# 					edge_labels.append(el)
# 		return edge_labels
#
#
# 	def _get_dataset_size(self):
# 		return len(self._graphs)
#
#
# 	def _get_all_node_nums(self):
# 		return [nx.number_of_nodes(G) for G in self._graphs]
#
#
# 	def _get_total_node_nums(self, all_node_nums):
# 		return np.sum(all_node_nums)
#
#
# 	def _get_ave_node_num(self, all_node_nums):
# 		return np.mean(all_node_nums)
#
#
# 	def _get_min_node_num(self, all_node_nums):
# 		return np.amin(all_node_nums)
#
#
# 	def _get_max_node_num(self, all_node_nums):
# 		return np.amax(all_node_nums)
#
#
# 	def _get_all_edge_nums(self):
# 		return [nx.number_of_edges(G) for G in self._graphs]
#
#
# 	def _get_total_edge_nums(self, all_edge_nums):
# 		return np.sum(all_edge_nums)
#
#
# 	def _get_ave_edge_num(self, all_edge_nums):
# 		return np.mean(all_edge_nums)
#
#
# 	def _get_min_edge_num(self, all_edge_nums):
# 		return np.amin(all_edge_nums)
#
#
# 	def _get_max_edge_num(self, all_edge_nums):
# 		return np.amax(all_edge_nums)
#
#
# 	def _get_node_label_dim(self):
# 		return len(self._node_labels)
#
#
# 	def _get_node_label_num(self, node_label):
# 		nl = set()
# 		for G in self._graphs:
# 			nl = nl | set(nx.get_node_attributes(G, node_label).values())
# 		return len(nl)
#
#
# 	def _get_edge_label_dim(self):
# 		return len(self._edge_labels)
#
#
# 	def _get_edge_label_num(self, edge_label):
# 		el = set()
# 		for G in self._graphs:
# 			el = el | set(nx.get_edge_attributes(G, edge_label).values())
# 		return len(el)
#
#
# 	def _is_directed(self):
# 		return nx.is_directed(self._graphs[0])
#
#
# 	def _get_all_node_degrees(self):
# 		return [np.mean(list(dict(G.degree()).values())) for G in self._graphs]
#
#
# 	def _get_ave_node_degree(self, all_node_degrees):
# 		return np.mean(all_node_degrees)
#
#
# 	def _get_max_node_degree(self, all_node_degrees):
# 		return np.amax(all_node_degrees)
#
#
# 	def _get_min_node_degree(self, all_node_degrees):
# 		return np.amin(all_node_degrees)
#
#
# 	def _get_all_fill_factors(self):
# 		"""Get fill factor, the number of non-zero entries in the adjacency matrix.

# 		Returns
# 		-------
# 		list[float]
# 			List of fill factors for all graphs.
# 		"""
# 		return [nx.number_of_edges(G) / (nx.number_of_nodes(G) ** 2) for G in self._graphs]
#

# 	def _get_ave_fill_factor(self, all_fill_factors):
# 		return np.mean(all_fill_factors)
#
#
# 	def _get_max_fill_factor(self, all_fill_factors):
# 		return np.amax(all_fill_factors)
#
#
# 	def _get_min_fill_factor(self, all_fill_factors):
# 		return np.amin(all_fill_factors)
#
#
# 	def _get_substructures(self):
# 		subs = set()
# 		for G in self._graphs:
# 			degrees = list(dict(G.degree()).values())
# 			if any(i == 2 for i in degrees):
# 				subs.add('linear')
# 			if np.amax(degrees) >= 3:
# 				subs.add('non linear')
# 			if 'linear' in subs and 'non linear' in subs:
# 				break

# 		if self._directed:
# 			for G in self._graphs:
# 				if len(list(nx.find_cycle(G))) > 0:
# 					subs.add('cyclic')
# 					break
# 			# else:
# 			#	 # @todo: this method does not work for big graph with large amount of edges like D&D, try a better way.
# 			#	 upper = np.amin([nx.number_of_edges(G) for G in Gn]) * 2 + 10
# 			#	 for G in Gn:
# 			#		 if (nx.number_of_edges(G) < upper):
# 			#			 cyc = list(nx.simple_cycles(G.to_directed()))
# 			#			 if any(len(i) > 2 for i in cyc):
# 			#				 subs.add('cyclic')
# 			#				 break
# 			#	 if 'cyclic' not in subs:
# 			#		 for G in Gn:
# 			#			 cyc = list(nx.simple_cycles(G.to_directed()))
# 			#			 if any(len(i) > 2 for i in cyc):
# 			#				 subs.add('cyclic')
# 			#				 break
#
# 			return subs
#
#
# 	def _get_class_num(self):
# 		return len(set(self._targets))
#
#
# 	def _get_node_attr_dim(self):
# 		return len(self._node_attrs)
#
#
# 	def _get_edge_attr_dim(self):
# 		return len(self._edge_attrs)

#
# 	def _compute_all_degree_entropy(self, base=None):
# 		"""Compute the entropy of degree distribution of each graph.

# 		Parameters
# 		----------
# 		base : float, optional
# 			The logarithmic base to use. The default is ``e`` (natural logarithm).

# 		Returns
# 		-------
# 		degree_entropy : float
# 			The calculated entropy.
# 		"""
# 		from gklearn.utils.stats import entropy
#
# 		degree_entropy = []
# 		for g in self._graphs:
# 			degrees = list(dict(g.degree()).values())
# 			en = entropy(degrees, base=base)
# 			degree_entropy.append(en)
# 		return degree_entropy
#
#
# 	@property
# 	def graphs(self):
# 		return self._graphs


# 	@property
# 	def targets(self):
# 		return self._targets
#
#
# 	@property
# 	def node_labels(self):
# 		return self._node_labels


# 	@property
# 	def edge_labels(self):
# 		return self._edge_labels
#
#
# 	@property
# 	def node_attrs(self):
# 		return self._node_attrs
#
#
# 	@property
# 	def edge_attrs(self):
# 		return self._edge_attrs
#
#
# def split_dataset_by_target(dataset):
# 	from gklearn.preimage.utils import get_same_item_indices
#
# 	graphs = dataset.graphs
# 	targets = dataset.targets
# 	datasets = []
# 	idx_targets = get_same_item_indices(targets)
# 	for key, val in idx_targets.items():
# 		sub_graphs = [graphs[i] for i in val]
# 		sub_dataset = Dataset()
# 		sub_dataset.load_graphs(sub_graphs, [key] * len(val))
# 		node_labels = dataset.node_labels.copy() if dataset.node_labels is not None else None
# 		node_attrs = dataset.node_attrs.copy() if dataset.node_attrs is not None else None
# 		edge_labels = dataset.edge_labels.copy() if dataset.edge_labels is not None else None
# 		edge_attrs = dataset.edge_attrs.copy() if dataset.edge_attrs is not None else None
# 		sub_dataset.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
# 		datasets.append(sub_dataset)
# 		# @todo: clean_labels?
# 	return datasets
