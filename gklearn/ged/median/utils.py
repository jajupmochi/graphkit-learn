#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:12:31 2020

@author: ljia
"""

def constant_node_costs(edit_cost_name):
	if edit_cost_name == 'NON_SYMBOLIC' or edit_cost_name == 'LETTER2' or edit_cost_name == 'LETTER':
		return False
	elif edit_cost_name == 'CONSTANT':
		return True
	else:
		raise Exception('Can not recognize the given edit cost. Possible edit costs include: "NON_SYMBOLIC", "LETTER", "LETTER2", "CONSTANT".')
#	 elif edit_cost_name != '':
# # 		throw ged::Error("Invalid dataset " + dataset + ". Usage: ./median_tests <AIDS|Mutagenicity|Letter-high|Letter-med|Letter-low|monoterpenoides|SYNTHETICnew|Fingerprint|COIL-DEL>");
#		 return False
	# return True
	
	
def mge_options_to_string(options):
	opt_str = ' '
	for key, val in options.items():
		if key == 'init_type':
			opt_str += '--init-type ' + str(val) + ' '
		elif key == 'random_inits':
			opt_str += '--random-inits ' + str(val) + ' '
		elif key == 'randomness':
			opt_str += '--randomness ' + str(val) + ' '
		elif key == 'verbose':
			opt_str += '--stdout ' + str(val) + ' '
		elif key == 'parallel':
			opt_str += '--parallel ' + ('TRUE' if val else 'FALSE') + ' '
		elif key == 'update_order':
			opt_str += '--update-order ' + ('TRUE' if val else 'FALSE') + ' '
		elif key == 'sort_graphs':
			opt_str += '--sort-graphs ' + ('TRUE' if val else 'FALSE') + ' '
		elif key == 'refine':
 			opt_str += '--refine ' + ('TRUE' if val else 'FALSE') + ' '
		elif key == 'time_limit':
 			opt_str += '--time-limit ' + str(val) + ' '
		elif key == 'max_itrs':
			opt_str += '--max-itrs ' + str(val) + ' '
		elif key == 'max_itrs_without_update':
			opt_str += '--max-itrs-without-update ' + str(val) + ' '
		elif key == 'seed':
			opt_str += '--seed ' + str(val) + ' '
		elif key == 'epsilon':
			opt_str += '--epsilon ' + str(val) + ' '
		elif key == 'inits_increase_order':
			opt_str += '--inits-increase-order ' + str(val) + ' '
		elif key == 'init_type_increase_order':
			opt_str += '--init-type-increase-order ' + str(val) + ' '
		elif key == 'max_itrs_increase_order':
			opt_str += '--max-itrs-increase-order ' + str(val) + ' '
# 		else:
# 			valid_options = '[--init-type <arg>] [--random_inits <arg>] [--randomness <arg>] [--seed <arg>] [--verbose <arg>] '
# 			valid_options += '[--time_limit <arg>] [--max_itrs <arg>] [--epsilon <arg>] '
# 			valid_options += '[--inits_increase_order <arg>] [--init_type_increase_order <arg>] [--max_itrs_increase_order <arg>]'
# 			raise Exception('Invalid option "' + key + '". Options available = "' + valid_options + '"')

	return opt_str