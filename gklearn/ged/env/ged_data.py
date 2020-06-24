#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:05:01 2020

@author: ljia
"""
from gklearn.ged.env import Options, OptionsStringMap
from gklearn.ged.edit_costs import Constant
from gklearn.utils import SpecialLabel, dummy_node


class GEDData(object):
	

	def __init__(self):
		self._graphs = []
		self._graph_names = []
		self._graph_classes = []
		self._num_graphs_without_shuffled_copies = 0
		self._strings_to_internal_node_ids = []
		self._internal_node_ids_to_strings = []
		self._edit_cost = None
		self._node_costs = None
		self._edge_costs = None
		self._node_labels = []
		self._edge_labels = []
		self._init_type = Options.InitType.EAGER_WITHOUT_SHUFFLED_COPIES
		self._delete_edit_cost = True
		self._max_num_nodes = 0
		self._max_num_edges = 0
		
		
	def num_graphs(self):
		"""
	/*!
	 * @brief Returns the number of graphs.
	 * @return Number of graphs in the instance.
	 */
		"""
		return len(self._graphs)
	
	
	def shuffled_graph_copies_available(self):
		"""
	/*!
	 * @brief Checks if shuffled graph copies are available.
	 * @return Boolean @p true if shuffled graph copies are available.
	 */
		"""
		return (self._init_type == Options.InitType.EAGER_WITH_SHUFFLED_COPIES or self._init_type == Options.InitType.LAZY_WITH_SHUFFLED_COPIES)
	
	
	def node_cost(self, label1, label2):
		"""
	/*!
	 * @brief Returns node relabeling, insertion, or deletion cost.
	 * @param[in] label1 First node label.
	 * @param[in] label2 Second node label.
	 * @return Node relabeling cost if @p label1 and @p label2 are both different from ged::dummy_label(),
	 * node insertion cost if @p label1 equals ged::dummy_label and @p label2 does not,
	 * node deletion cost if @p label1 does not equal ged::dummy_label and @p label2 does,
	 * and 0 otherwise.
	 */
		"""
		if self._eager_init(): # @todo: check if correct
			return self._node_costs[label1, label2]
		if label1 == label2:
			return 0
		if label1 == SpecialLabel.DUMMY: # @todo: check dummy
			return self._edit_cost.node_ins_cost_fun(label2) # self._node_labels[label2 - 1]) # @todo: check
		if label2 == SpecialLabel.DUMMY: # @todo: check dummy
			return self._edit_cost.node_del_cost_fun(label1) # self._node_labels[label1 - 1])
		return self._edit_cost.node_rel_cost_fun(label1, label2) # self._node_labels[label1 - 1], self._node_labels[label2 - 1])
	
	
	def edge_cost(self, label1, label2):
		"""
	/*!
	 * @brief Returns edge relabeling, insertion, or deletion cost.
	 * @param[in] label1 First edge label.
	 * @param[in] label2 Second edge label.
	 * @return Edge relabeling cost if @p label1 and @p label2 are both different from ged::dummy_label(),
	 * edge insertion cost if @p label1 equals ged::dummy_label and @p label2 does not,
	 * edge deletion cost if @p label1 does not equal ged::dummy_label and @p label2 does,
	 * and 0 otherwise.
	 */
		"""
		if self._eager_init(): # @todo: check if correct
			return self._node_costs[label1, label2]
		if label1 == label2:
			return 0
		if label1 == SpecialLabel.DUMMY:
			return self._edit_cost.edge_ins_cost_fun(label2) # self._edge_labels[label2 - 1])
		if label2 == SpecialLabel.DUMMY:
			return self._edit_cost.edge_del_cost_fun(label1) # self._edge_labels[label1 - 1])
		return self._edit_cost.edge_rel_cost_fun(label1, label2) # self._edge_labels[label1 - 1], self._edge_labels[label2 - 1])
	
	
	def compute_induced_cost(self, g, h, node_map):
		"""
	/*!
	 * @brief Computes the edit cost between two graphs induced by a node map.
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[in,out] node_map Node map whose induced edit cost is to be computed.
	 */
		"""
		cost = 0
		
		# collect node costs
		for node in g.nodes():
			image = node_map.image(node)
			label2 = (SpecialLabel.DUMMY if image == dummy_node() else h.nodes[image]['label'])
			cost += self.node_cost(g.nodes[node]['label'], label2)
		for node in h.nodes():
			pre_image = node_map.pre_image(node)
			if pre_image == dummy_node():
				cost += self.node_cost(SpecialLabel.DUMMY, h.nodes[node]['label'])
				
		# collect edge costs
		for (n1, n2) in g.edges():
			image1 = node_map.image(n1)
			image2 = node_map.image(n2)
			label2 = (h.edges[(image2, image1)]['label'] if h.has_edge(image2, image1) else SpecialLabel.DUMMY)
			cost += self.edge_cost(g.edges[(n1, n2)]['label'], label2)
		for (n1, n2) in h.edges():
			if not g.has_edge(node_map.pre_image(n2), node_map.pre_image(n1)):
				cost += self.edge_cost(SpecialLabel.DUMMY, h.edges[(n1, n2)]['label'])
				
		node_map.set_induced_cost(cost)
			
			
	def _set_edit_cost(self, edit_cost, edit_cost_constants):
		if self._delete_edit_cost:
			self._edit_cost = None
			
		if isinstance(edit_cost, str):
			edit_cost = OptionsStringMap.EditCosts[edit_cost]
			
		if edit_cost == Options.EditCosts.CHEM_1:
			if len(edit_cost_constants) == 4:
				self._edit_cost = CHEM1(edit_cost_constants[0], edit_cost_constants[1], edit_cost_constants[2], edit_cost_constants[3])
			elif len(edit_cost_constants) == 0:
				self._edit_cost = CHEM1()
			else:
				raise Exception('Wrong number of constants for selected edit costs Options::EditCosts::CHEM_1. Expected: 4 or 0; actual:', len(edit_cost_constants), '.')
		elif edit_cost == Options.EditCosts.LETTER:
			if len(edit_cost_constants) == 3:
				self._edit_cost = Letter(edit_cost_constants[0], edit_cost_constants[1], edit_cost_constants[2])
			elif len(edit_cost_constants) == 0:
				self._edit_cost = Letter()
			else:
				raise Exception('Wrong number of constants for selected edit costs Options::EditCosts::LETTER. Expected: 3 or 0; actual:', len(edit_cost_constants), '.')
		elif edit_cost == Options.EditCosts.LETTER2:
			if len(edit_cost_constants) == 5:
				self._edit_cost = Letter2(edit_cost_constants[0], edit_cost_constants[1], edit_cost_constants[2], edit_cost_constants[3], edit_cost_constants[4])
			elif len(edit_cost_constants) == 0:
				self._edit_cost = Letter2()
			else:
				raise Exception('Wrong number of constants for selected edit costs Options::EditCosts::LETTER2. Expected: 5 or 0; actual:', len(edit_cost_constants), '.')
		elif edit_cost == Options.EditCosts.NON_SYMBOLIC:
			if len(edit_cost_constants) == 6:
				self._edit_cost = NonSymbolic(edit_cost_constants[0], edit_cost_constants[1], edit_cost_constants[2], edit_cost_constants[3], edit_cost_constants[4], edit_cost_constants[5])
			elif len(edit_cost_constants) == 0:
				self._edit_cost = NonSymbolic()
			else:
				raise Exception('Wrong number of constants for selected edit costs Options::EditCosts::NON_SYMBOLIC. Expected: 6 or 0; actual:', len(edit_cost_constants), '.')
		elif edit_cost == Options.EditCosts.CONSTANT:
			if len(edit_cost_constants) == 6:
				self._edit_cost = Constant(edit_cost_constants[0], edit_cost_constants[1], edit_cost_constants[2], edit_cost_constants[3], edit_cost_constants[4], edit_cost_constants[5])
			elif len(edit_cost_constants) == 0:
				self._edit_cost = Constant()
			else:
				raise Exception('Wrong number of constants for selected edit costs Options::EditCosts::CONSTANT. Expected: 6 or 0; actual:', len(edit_cost_constants), '.')
				
		self._delete_edit_cost = True
		
		
	def _eager_init(self):
		return (self._init_type == Options.InitType.EAGER_WITHOUT_SHUFFLED_COPIES or self._init_type == Options.InitType.EAGER_WITH_SHUFFLED_COPIES)