#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:37:36 2020

@author: ljia
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class LSAPESolver(object):
	
	
	def __init__(self, cost_matrix=None):
		"""
	/*!
	 * @brief Constructs solver for LSAPE problem instance.
	 * @param[in] cost_matrix Pointer to the LSAPE problem instance that should be solved.
	 */
		"""
		self.__cost_matrix = cost_matrix
		self.__model = 'ECBP'
		self.__greedy_method = 'BASIC'
		self.__solve_optimally = True
		self.__minimal_cost = 0
		self.__row_to_col_assignments = []
		self.__col_to_row_assignments = []
		self.__dual_var_rows = [] # @todo
		self.__dual_var_cols = [] # @todo
		
	
	def clear_solution(self):
		"""Clears a previously computed solution.
		"""
		self.__minimal_cost = 0
		self.__row_to_col_assignments.clear()
		self.__col_to_row_assignments.clear()
		self.__row_to_col_assignments.append([]) # @todo
		self.__col_to_row_assignments.append([])
		self.__dual_var_rows = [] # @todo
		self.__dual_var_cols = [] # @todo
		
		
	def set_model(self, model):
		"""
	/*!
	 * @brief Makes the solver use a specific model for optimal solving.
	 * @param[in] model The model that should be used.
	 */
		"""
		self.__solve_optimally = True
		self.__model = model
		
	
	def solve(self, num_solutions=1):
		"""
	/*!
	 * @brief Solves the LSAPE problem instance.
	 * @param[in] num_solutions The maximal number of solutions that should be computed.
	 */
		"""
		self.clear_solution()
		if self.__solve_optimally:
			row_ind, col_ind = linear_sum_assignment(self.__cost_matrix) # @todo: only hungarianLSAPE ('ECBP') can be used.
			self.__row_to_col_assignments[0] = col_ind
			self.__col_to_row_assignments[0] = np.argsort(col_ind) # @todo: might be slow, can use row_ind
			self.__compute_cost_from_assignments()
			if num_solutions > 1:
				pass # @todo:
		else:
			print('here is non op.')
			pass # @todo: greedy.
# 			self.__

	
	def minimal_cost(self):
		"""
	/*!
	 * @brief Returns the cost of the computed solutions.
	 * @return Cost of computed solutions.
	 */
		"""
		return self.__minimal_cost
	
	
	def get_assigned_col(self, row, solution_id=0):
		"""
	/*!
	 * @brief Returns the assigned column.
	 * @param[in] row Row whose assigned column should be returned.
	 * @param[in] solution_id ID of the solution where the assignment should be looked up.
	 * @returns Column to which @p row is assigned to in solution with ID @p solution_id or ged::undefined() if @p row is not assigned to any column.
	 */
		"""
		return self.__row_to_col_assignments[solution_id][row]
		
		
	def get_assigned_row(self, col, solution_id=0):
		"""
	/*!
	 * @brief Returns the assigned row.
	 * @param[in] col Column whose assigned row should be returned.
	 * @param[in] solution_id ID of the solution where the assignment should be looked up.
	 * @returns Row to which @p col is assigned to in solution with ID @p solution_id or ged::undefined() if @p col is not assigned to any row.
	 */
		"""
		return self.__col_to_row_assignments[solution_id][col]
		
		
	def num_solutions(self):
		"""
	/*!
	 * @brief Returns the number of solutions.
	 * @returns Actual number of solutions computed by solve(). Might be smaller than @p num_solutions.
	 */
		"""
		return len(self.__row_to_col_assignments)


	def __compute_cost_from_assignments(self): # @todo
		self.__minimal_cost = np.sum(self.__cost_matrix[range(0, len(self.__row_to_col_assignments[0])), self.__row_to_col_assignments[0]])