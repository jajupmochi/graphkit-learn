#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:49:24 2020

@author: ljia
"""


class EditCost(object):
	
	
	def __init__(self):
		pass
	
	
	def node_ins_cost_fun(self, node_label):
		"""
	/*!
	 * @brief Node insertions cost function.
	 * @param[in] node_label A node label.
	 * @return The cost of inserting a node with label @p node_label.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0

	
	def node_del_cost_fun(self, node_label):
		"""
	/*!
	 * @brief Node deletion cost function.
	 * @param[in] node_label A node label.
	 * @return The cost of deleting a node with label @p node_label.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0

	
	def node_rel_cost_fun(self, node_label_1, node_label_2):
		"""
	/*!
	 * @brief Node relabeling cost function.
	 * @param[in] node_label_1 A node label.
	 * @param[in] node_label_2 A node label.
	 * @return The cost of changing a node's label from @p node_label_1 to @p node_label_2.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0

	
	def edge_ins_cost_fun(self, edge_label):
		"""
	/*!
	 * @brief Edge insertion cost function.
	 * @param[in] edge_label An edge label.
	 * @return The cost of inserting an edge with label @p edge_label.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0

	
	def edge_del_cost_fun(self, edge_label):
		"""
	/*!
	 * @brief Edge deletion cost function.
	 * @param[in] edge_label An edge label.
	 * @return The cost of deleting an edge with label @p edge_label.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0

	
	def edge_rel_cost_fun(self, edge_label_1, edge_label_2):
		"""
	/*!
	 * @brief Edge relabeling cost function.
	 * @param[in] edge_label_1 An edge label.
	 * @param[in] edge_label_2 An edge label.
	 * @return The cost of changing an edge's label from @p edge_label_1 to @p edge_label_2.
	 * @note Must be implemented by derived classes of ged::EditCosts.
	 */		
		"""
		return 0