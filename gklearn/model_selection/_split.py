#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:13:26 2022

@author: ljia

Reference: scikit-learn.
"""
from abc import abstractmethod
import numbers
import warnings
import numpy as np
from sklearn.utils import check_random_state, check_array, column_or_1d, indexable
from sklearn.utils.validation import _num_samples
from sklearn.utils.multiclass import type_of_target


class BaseCrossValidatorWithValid(object):
	"""Base class for all cross-validators.
	Implementations must define `_iter_valid_test_masks` or `_iter_valid_stest_indices`.
	"""

	def split(self, X, y=None, groups=None):
		"""Generate indices to split data into training, valid, and test set.

		Parameters
		----------

		X : array-like of shape (n_samples, n_features)
			Training data, where `n_samples` is the number of samples
			and `n_features` is the number of features.

		y : array-like of shape (n_samples,)
			The target variable for supervised learning problems.

		groups : array-like of shape (n_samples,), default=None
			Group labels for the samples used while splitting the dataset into
			train/test set.

		Yields
		------
		train : ndarray
			The training set indices for that split.

		valid : ndarray
			The valid set indices for that split.

		test : ndarray
			The testing set indices for that split.
		"""
		X, y, groups = indexable(X, y, groups)
		indices = np.arange(_num_samples(X))
		for valid_index, test_index in self._iter_valid_test_masks(X, y, groups):
			train_index = indices[np.logical_not(np.logical_or(valid_index, test_index))]
			valid_index = indices[valid_index]
			test_index = indices[test_index]
			yield train_index, valid_index, test_index


	# Since subclasses must implement either _iter_valid_test_masks or
	# _iter_valid_test_indices, neither can be abstract.
	def _iter_valid_test_masks(self, X=None, y=None, groups=None):
		"""Generates boolean masks corresponding to valid and test sets.
		By default, delegates to _iter_valid_test_indices(X, y, groups)
		"""
		for valid_index, test_index in self._iter_valid_test_indices(X, y, groups):
			valid_mask = np.zeros(_num_samples(X), dtype=bool)
			test_mask = np.zeros(_num_samples(X), dtype=bool)
			valid_mask[valid_index] = True
			test_mask[test_index] = True
			yield valid_mask, test_mask


	def _iter_valid_test_indices(self, X=None, y=None, groups=None):
		"""Generates integer indices corresponding to valid and test sets."""
		raise NotImplementedError


	@abstractmethod
	def get_n_splits(self, X=None, y=None, groups=None):
		"""Returns the number of splitting iterations in the cross-validator"""


	def __repr__(self):
		return _build_repr(self)


class _BaseKFoldWithValid(BaseCrossValidatorWithValid):
	"""Base class for KFoldWithValid, GroupKFoldWithValid, and StratifiedKFoldWithValid"""

	@abstractmethod
	def __init__(self, n_splits, *, stratify, shuffle, random_state):
		if not isinstance(n_splits, numbers.Integral):
			raise ValueError(
				'The number of folds must be of Integral type. '
				'%s of type %s was passed.' % (n_splits, type(n_splits))
			)
		n_splits = int(n_splits)

		if n_splits <= 2:
			raise ValueError(
				'k-fold cross-validation requires at least one'
				' train/valid/test split by setting n_splits=3 or more,'
				' got n_splits={0}.'.format(n_splits)
			)

		if not isinstance(shuffle, bool):
			raise TypeError('shuffle must be True or False; got {0}'.format(shuffle))

		if not shuffle and random_state is not None:  # None is the default
			raise ValueError(
				'Setting a random_state has no effect since shuffle is '
				'False. You should leave '
				'random_state to its default (None), or set shuffle=True.',
			)

		self.n_splits = n_splits
		self.stratify = stratify
		self.shuffle = shuffle
		self.random_state = random_state


	def split(self, X, y=None, groups=None):
		"""Generate indices to split data into training, valid and test set."""
		X, y, groups = indexable(X, y, groups)
		n_samples = _num_samples(X)
		if self.n_splits > n_samples:
			raise ValueError(
				(
				 'Cannot have number of splits n_splits={0} greater'
				 ' than the number of samples: n_samples={1}.'
				 ).format(self.n_splits, n_samples)
			)

		for train, valid, test in super().split(X, y, groups):
			yield train, valid, test


class KFoldWithValid(_BaseKFoldWithValid):


	def __init__(
			self,
			n_splits=5,
			*,
			stratify=False,
			shuffle=False,
			random_state=None
			):
		super().__init__(
			n_splits=n_splits,
			stratify=stratify,
			shuffle=shuffle,
			random_state=random_state
			)


	def _make_valid_test_folds(self, X, y=None):
		rng = check_random_state(self.random_state)
		y = np.asarray(y)
		type_of_target_y = type_of_target(y)
		allowed_target_types = ('binary', 'multiclass')
		if type_of_target_y not in allowed_target_types:
			raise ValueError(
				'Supported target types are: {}. Got {!r} instead.'.format(
					allowed_target_types, type_of_target_y
				)
			)

		y = column_or_1d(y)

		_, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
		# y_inv encodes y according to lexicographic order. We invert y_idx to
		# map the classes so that they are encoded by order of appearance:
		# 0 represents the first label appearing in y, 1 the second, etc.
		_, class_perm = np.unique(y_idx, return_inverse=True)
		y_encoded = class_perm[y_inv]

		n_classes = len(y_idx)
		y_counts = np.bincount(y_encoded)
		min_groups = np.min(y_counts)
		if np.all(self.n_splits > y_counts):
			raise ValueError(
				"n_splits=%d cannot be greater than the"
				" number of members in each class." % (self.n_splits)
			)
		if self.n_splits > min_groups:
			warnings.warn(
				"The least populated class in y has only %d"
				" members, which is less than n_splits=%d."
				% (min_groups, self.n_splits),
				UserWarning,
			)

		# Determine the optimal number of samples from each class in each fold,
		# using round robin over the sorted y. (This can be done direct from
		# counts, but that code is unreadable.)
		y_order = np.sort(y_encoded)
		allocation = np.asarray(
			[
				np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
				for i in range(self.n_splits)
			]
		)

		# To maintain the data order dependencies as best as possible within
		# the stratification constraint, we assign samples from each class in
		# blocks (and then mess that up when shuffle=True).
		test_folds = np.empty(len(y), dtype='i')
		for k in range(n_classes):
			# since the kth column of allocation stores the number of samples
			# of class k in each test set, this generates blocks of fold
			# indices corresponding to the allocation for class k.
			folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
			if self.shuffle:
				rng.shuffle(folds_for_class)
			test_folds[y_encoded == k] = folds_for_class
		return test_folds


	def _iter_valid_test_masks(self, X, y=None, groups=None):
		test_folds = self._make_valid_test_folds(X, y)
		for i in range(self.n_splits):
			if i + 1 < self.n_splits:
				j = i + 1
			else:
				j = 0
			yield test_folds == i, test_folds == j


	def split(self, X, y, groups=None):
		y = check_array(y, input_name='y', ensure_2d=False, dtype=None)
		return super().split(X, y, groups)


class _RepeatedSplitsWithValid(object):


	def __init__(
			self,
			cv,
			*,
			n_repeats=10,
			random_state=None,
			**cvargs
			):
		if not isinstance(n_repeats, int):
			raise ValueError('Number of repetitions must be of integer type.')

		if n_repeats <= 0:
			raise ValueError('Number of repetitions must be greater than 0.')

		self.cv = cv
		self.n_repeats = n_repeats
		self.random_state = random_state
		self.cvargs = cvargs


	def split(self, X, y=None, groups=None):
		n_repeats = self.n_repeats
		rng = check_random_state(self.random_state)

		for idx in range(n_repeats):
			cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
			for train_index, valid_index, test_index in cv.split(X, y, groups):
				yield train_index, valid_index, test_index


class RepeatedKFoldWithValid(_RepeatedSplitsWithValid):


	def __init__(
			self,
			*,
			n_splits=5,
			n_repeats=10,
			stratify=False,
			random_state=None
			):
		super().__init__(
			KFoldWithValid,
			n_repeats=n_repeats,
			stratify=stratify,
			random_state=random_state,
			n_splits=n_splits,
			)