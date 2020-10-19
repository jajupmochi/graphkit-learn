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
		self._cost_matrix = cost_matrix
		self._model = 'ECBP'
		self._greedy_method = 'BASIC'
		self._solve_optimally = True
		self._minimal_cost = 0
		self._row_to_col_assignments = []
		self._col_to_row_assignments = []
		self._dual_var_rows = [] # @todo
		self._dual_var_cols = [] # @todo
		
	
	def clear_solution(self):
		"""Clears a previously computed solution.
		"""
		self._minimal_cost = 0
		self._row_to_col_assignments.clear()
		self._col_to_row_assignments.clear()
		self._row_to_col_assignments.append([]) # @todo
		self._col_to_row_assignments.append([])
		self._dual_var_rows = [] # @todo
		self._dual_var_cols = [] # @todo
		
		
	def set_model(self, model):
		"""
	/*!
	 * @brief Makes the solver use a specific model for optimal solving.
	 * @param[in] model The model that should be used.
	 */
		"""
		self._solve_optimally = True
		self._model = model
		
	
	def solve(self, num_solutions=1):
		"""
	/*!
	 * @brief Solves the LSAPE problem instance.
	 * @param[in] num_solutions The maximal number of solutions that should be computed.
	 */
		"""
		self.clear_solution()
		if self._solve_optimally:
			row_ind, col_ind = linear_sum_assignment(self._cost_matrix) # @todo: only hungarianLSAPE ('ECBP') can be used.
			self._row_to_col_assignments[0] = col_ind
			self._col_to_row_assignments[0] = np.argsort(col_ind) # @todo: might be slow, can use row_ind
			self._compute_cost_from_assignments()
			if num_solutions > 1:
				pass # @todo:
		else:
			print('here is non op.')
			pass # @todo: greedy.
# 			self._

	
	def minimal_cost(self):
		"""
	/*!
	 * @brief Returns the cost of the computed solutions.
	 * @return Cost of computed solutions.
	 */
		"""
		return self._minimal_cost
	
	
	def get_assigned_col(self, row, solution_id=0):
		"""
	/*!
	 * @brief Returns the assigned column.
	 * @param[in] row Row whose assigned column should be returned.
	 * @param[in] solution_id ID of the solution where the assignment should be looked up.
	 * @returns Column to which @p row is assigned to in solution with ID @p solution_id or ged::undefined() if @p row is not assigned to any column.
	 */
		"""
		return self._row_to_col_assignments[solution_id][row]
		
		
	def get_assigned_row(self, col, solution_id=0):
		"""
	/*!
	 * @brief Returns the assigned row.
	 * @param[in] col Column whose assigned row should be returned.
	 * @param[in] solution_id ID of the solution where the assignment should be looked up.
	 * @returns Row to which @p col is assigned to in solution with ID @p solution_id or ged::undefined() if @p col is not assigned to any row.
	 */
		"""
		return self._col_to_row_assignments[solution_id][col]
		
		
	def num_solutions(self):
		"""
	/*!
	 * @brief Returns the number of solutions.
	 * @returns Actual number of solutions computed by solve(). Might be smaller than @p num_solutions.
	 */
		"""
		return len(self._row_to_col_assignments)


	def _compute_cost_from_assignments(self): # @todo
		self._minimal_cost = np.sum(self._cost_matrix[range(0, len(self._row_to_col_assignments[0])), self._row_to_col_assignments[0]])