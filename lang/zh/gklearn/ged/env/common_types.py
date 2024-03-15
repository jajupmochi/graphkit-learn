#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:17:38 2020

@author: ljia
"""

from enum import Enum, unique


class Options(object):
	"""Contains enums for options employed by ged::GEDEnv.
	"""


	@unique
	class GEDMethod(Enum):
		"""Selects the method.
		"""
# @todo: what is this? #ifdef GUROBI
		F1 = 1                  # Selects ged::F1.
		F2 = 2                 # Selects ged::F2.
		COMPACT_MIP = 3         # Selects ged::CompactMIP.
		BLP_NO_EDGE_LABELS = 4  # Selects ged::BLPNoEdgeLabels.
#endif /* GUROBI */
		BRANCH = 5              # Selects ged::Branch.
		BRANCH_FAST = 6         # Selects ged::BranchFast.
		BRANCH_TIGHT = 7        # Selects ged::BranchTight.
		BRANCH_UNIFORM = 8      # Selects ged::BranchUniform.
		BRANCH_COMPACT = 9      # Selects ged::BranchCompact.
		PARTITION = 10           # Selects ged::Partition.
		HYBRID = 11              # Selects ged::Hybrid.
		RING = 12                # Selects ged::Ring.
		ANCHOR_AWARE_GED = 13    # Selects ged::AnchorAwareGED.
		WALKS = 14               # Selects ged::Walks.
		IPFP = 15                # Selects ged::IPFP
		BIPARTITE = 16           # Selects ged::Bipartite.
		SUBGRAPH = 17            # Selects ged::Subgraph.
		NODE = 18                # Selects ged::Node.
		RING_ML = 19             # Selects ged::RingML.
		BIPARTITE_ML = 20        # Selects ged::BipartiteML.
		REFINE = 21              # Selects ged::Refine.
		BP_BEAM = 22             # Selects ged::BPBeam.
		SIMULATED_ANNEALING = 23 # Selects ged::SimulatedAnnealing.
		HED = 24				 # Selects ged::HED.
		STAR = 25				 # Selects ged::Star.


	@unique
	class EditCosts(Enum):
		"""Selects the edit costs.
		"""
		CHEM_1 = 1      # Selects ged::CHEM1.
		CHEM_2 = 2      # Selects ged::CHEM2.
		CMU = 3         # Selects ged::CMU.
		GREC_1 = 4      # Selects ged::GREC1.
		GREC_2 = 5      # Selects ged::GREC2.
		PROTEIN = 6     # Selects ged::Protein.
		FINGERPRINT = 7 # Selects ged::Fingerprint.
		LETTER = 8      # Selects ged::Letter.
		LETTER2 = 9     # Selects ged:Letter2.
		NON_SYMBOLIC = 10 # Selects ged:NonSymbolic.
		CONSTANT = 11    # Selects ged::Constant.
		
		
	@unique
	class InitType(Enum):
		"""@brief Selects the initialization type of the environment.
		* @details If eager initialization is selected, all edit costs are pre-computed when initializing the environment.
		* Otherwise, they are computed at runtime. If initialization with shuffled copies is selected, shuffled copies of
		* all graphs are created. These copies are used when calling ged::GEDEnv::run_method() with two identical graph IDs.
		* In this case, one of the IDs is internally replaced by the ID of the shuffled copy and the graph is hence
		* compared to an isomorphic but non-identical graph. If initialization without shuffled copies is selected, no shuffled copies
		* are created and calling ged::GEDEnv::run_method() with two identical graph IDs amounts to comparing a graph to itself.
		"""
		LAZY_WITHOUT_SHUFFLED_COPIES = 1  # Lazy initialization, no shuffled graph copies are constructed.
		EAGER_WITHOUT_SHUFFLED_COPIES = 2 # Eager initialization, no shuffled graph copies are constructed.
		LAZY_WITH_SHUFFLED_COPIES = 3     # Lazy initialization, shuffled graph copies are constructed.
		EAGER_WITH_SHUFFLED_COPIES = 4    # Eager initialization, shuffled graph copies are constructed.	
	
	
	@unique
	class AlgorithmState(Enum):
		"""can be used to specify the state of an algorithm.
		"""
		CALLED = 1 # The algorithm has been called.
		INITIALIZED = 2 # The algorithm has been initialized.
		CONVERGED = 3 # The algorithm has converged.
		TERMINATED = 4 # The algorithm has terminated.	


class OptionsStringMap(object):
	
	
	# Map of available computation methods between enum type and string. 
	GEDMethod = {
		"BRANCH": Options.GEDMethod.BRANCH,
		"BRANCH_FAST": Options.GEDMethod.BRANCH_FAST,
		"BRANCH_TIGHT": Options.GEDMethod.BRANCH_TIGHT,
		"BRANCH_UNIFORM": Options.GEDMethod.BRANCH_UNIFORM,
		"BRANCH_COMPACT": Options.GEDMethod.BRANCH_COMPACT,
		"PARTITION": Options.GEDMethod.PARTITION,
		"HYBRID": Options.GEDMethod.HYBRID,
		"RING": Options.GEDMethod.RING,
		"ANCHOR_AWARE_GED": Options.GEDMethod.ANCHOR_AWARE_GED,
		"WALKS": Options.GEDMethod.WALKS,
		"IPFP": Options.GEDMethod.IPFP,
		"BIPARTITE": Options.GEDMethod.BIPARTITE,
		"SUBGRAPH": Options.GEDMethod.SUBGRAPH,
		"NODE": Options.GEDMethod.NODE,
		"RING_ML": Options.GEDMethod.RING_ML,
		"BIPARTITE_ML": Options.GEDMethod.BIPARTITE_ML,
		"REFINE": Options.GEDMethod.REFINE,
		"BP_BEAM": Options.GEDMethod.BP_BEAM,
		"SIMULATED_ANNEALING": Options.GEDMethod.SIMULATED_ANNEALING,
		"HED": Options.GEDMethod.HED,
		"STAR": Options.GEDMethod.STAR,
		# ifdef GUROBI
		"F1": Options.GEDMethod.F1,
		"F2": Options.GEDMethod.F2,
		"COMPACT_MIP": Options.GEDMethod.COMPACT_MIP,
		"BLP_NO_EDGE_LABELS": Options.GEDMethod.BLP_NO_EDGE_LABELS
	}

	
	# Map of available edit cost functions between enum type and string.
	EditCosts = {
		"CHEM_1": Options.EditCosts.CHEM_1,
		"CHEM_2": Options.EditCosts.CHEM_2,
		"CMU": Options.EditCosts.CMU,
		"GREC_1": Options.EditCosts.GREC_1,
		"GREC_2": Options.EditCosts.GREC_2,
		"LETTER": Options.EditCosts.LETTER,
		"LETTER2": Options.EditCosts.LETTER2,
		"NON_SYMBOLIC": Options.EditCosts.NON_SYMBOLIC,
		"FINGERPRINT": Options.EditCosts.FINGERPRINT,
		"PROTEIN": Options.EditCosts.PROTEIN,
		"CONSTANT": Options.EditCosts.CONSTANT
	}
	
	# Map of available initialization types of the environment between enum type and string.
	InitType = {
		"LAZY_WITHOUT_SHUFFLED_COPIES": Options.InitType.LAZY_WITHOUT_SHUFFLED_COPIES,
		"EAGER_WITHOUT_SHUFFLED_COPIES": Options.InitType.EAGER_WITHOUT_SHUFFLED_COPIES,
		"LAZY_WITH_SHUFFLED_COPIES": Options.InitType.LAZY_WITH_SHUFFLED_COPIES,
		"LAZY_WITH_SHUFFLED_COPIES": Options.InitType.LAZY_WITH_SHUFFLED_COPIES
	}
	

@unique
class AlgorithmState(Enum):
	"""can be used to specify the state of an algorithm.
	"""
	CALLED = 1 # The algorithm has been called.
	INITIALIZED = 2 # The algorithm has been initialized.
	CONVERGED = 3 # The algorithm has converged.
	TERMINATED = 4 # The algorithm has terminated.
	
