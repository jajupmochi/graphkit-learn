#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:13:56 2020

@author: ljia
"""
from gklearn.utils import dummy_node


def construct_node_map_from_solver(solver, node_map, solution_id):
	node_map.clear()
	num_nodes_g = node_map.num_source_nodes()
	num_nodes_h = node_map.num_target_nodes()
	
	# add deletions and substitutions
	for row in range(0, num_nodes_g):
		col = solver.get_assigned_col(row, solution_id)
		if col >= num_nodes_h:
			node_map.add_assignment(row, dummy_node())
		else:
			node_map.add_assignment(row, col)
			
	# insertions.
	for col in range(0, num_nodes_h):
		if solver.get_assigned_row(col, solution_id) >= num_nodes_g:
			node_map.add_assignment(dummy_node(), col)
	

def options_string_to_options_map(options_string):
    """Transforms an options string into an options map.
    
    Parameters
    ----------
    options_string : string
        Options string of the form "[--<option> <arg>] [...]".
    
    Return
    ------
    options_map : dict{string : string}
        Map with one key-value pair (<option>, <arg>) for each option contained in the string.
    """
    if options_string == '':
        return
    options_map = {}
    words = []
    tokenize(options_string, ' ', words)
    expect_option_name = True
    for word in words:
        if expect_option_name:
            is_opt_name, word = is_option_name(word)
            if is_opt_name:
                option_name = word
                if option_name in options_map:
                    raise Exception('Multiple specification of option "' + option_name + '".')
                options_map[option_name] = ''
            else:
                raise Exception('Invalid options "' + options_string + '". Usage: options = "[--<option> <arg>] [...]"')
        else:
            is_opt_name, word = is_option_name(word)
            if is_opt_name:
                raise Exception('Invalid options "' + options_string + '". Usage: options = "[--<option> <arg>] [...]"')
            else:
                options_map[option_name] = word
        expect_option_name = not expect_option_name
    return options_map
    

def tokenize(sentence, sep, words):
    """Separates a sentence into words separated by sep (unless contained in single quotes).
    
    Parameters
    ----------
    sentence : string
        The sentence that should be tokenized.
        
    sep : string 
        The separator. Must be different from "'".
        
    words : list[string]
        The obtained words.
    """
    outside_quotes = True
    word_length = 0
    pos_word_start = 0
    for pos in range(0, len(sentence)):
        if sentence[pos] == '\'':
            if not outside_quotes and pos < len(sentence) - 1:
                if sentence[pos + 1] != sep:
                    raise Exception('Sentence contains closing single quote which is followed by a char different from ' + sep + '.')
            word_length += 1
            outside_quotes = not outside_quotes
        elif outside_quotes and sentence[pos] == sep:
            if word_length > 0:
                words.append(sentence[pos_word_start:pos_word_start + word_length])
            pos_word_start = pos + 1
            word_length = 0
        else:
            word_length += 1
    if not outside_quotes:
        raise Exception('Sentence contains unbalanced single quotes.')
    if word_length > 0:
        words.append(sentence[pos_word_start:pos_word_start + word_length])


def is_option_name(word):
    """Checks whether a word is an option name and, if so, removes the leading dashes.
    
    Parameters
    ----------
    word : string
        Word.
        
    return
    ------
    True if word is of the form "--<option>".
    
    word : string
        The word without the leading dashes.
    """
    if word[0] == '\'':
        word = word[1:len(word) - 2]
        return False, word
    if len(word) < 3:
        return False, word
    if word[0] == '-' and word[1] == '-' and word[2] != '-':
        word = word[2:]
        return True, word
    return False, word