#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:09:04 2020

@author: ljia
"""
import re

def convert_function(cpp_code):
# f_cpp = open('cpp_code.cpp', 'r')
# # f_cpp = open('cpp_ext/src/median_graph_estimator.ipp', 'r')
# 	cpp_code = f_cpp.read()
	python_code = cpp_code.replace('else if (', 'elif ')
	python_code = python_code.replace('if (', 'if ')
	python_code = python_code.replace('else {', 'else:')
	python_code = python_code.replace(') {', ':')
	python_code = python_code.replace(';\n', '\n')
	python_code = re.sub('\n(.*)}\n', '\n\n', python_code)
	# python_code = python_code.replace('}\n', '')
	python_code = python_code.replace('throw', 'raise')
	python_code = python_code.replace('error', 'Exception')
	python_code = python_code.replace('"', '\'')
	python_code = python_code.replace('\\\'', '"')
	python_code = python_code.replace('try {', 'try:')
	python_code = python_code.replace('true', 'True')
	python_code = python_code.replace('false', 'False')
	python_code = python_code.replace('catch (...', 'except')
	# python_code = re.sub('std::string\(\'(.*)\'\)', '$1', python_code)
	
	return python_code



# # python_code = python_code.replace('}\n', '')




# python_code = python_code.replace('option.first', 'opt_name')
# python_code = python_code.replace('option.second', 'opt_val')
# python_code = python_code.replace('ged::Error', 'Exception')
# python_code = python_code.replace('std::string(\'Invalid argument "\')', '\'Invalid argument "\'')


# f_cpp.close()
# f_python = open('python_code.py', 'w')
# f_python.write(python_code)
# f_python.close()


def convert_function_comment(cpp_fun_cmt, param_types):
	cpp_fun_cmt = cpp_fun_cmt.replace('\t', '')
	cpp_fun_cmt = cpp_fun_cmt.replace('\n * ', ' ')
	# split the input comment according to key words.
	param_split = None
	note = None
	cmt_split = cpp_fun_cmt.split('@brief')[1]
	brief = cmt_split
	if '@param' in cmt_split:
		cmt_split = cmt_split.split('@param')
		brief = cmt_split[0]
		param_split = cmt_split[1:]
	if '@note' in cmt_split[-1]:
		note_split = cmt_split[-1].split('@note')
		if param_split is not None:
			param_split.pop()
			param_split.append(note_split[0])
		else:
			brief = note_split[0]
		note = note_split[1]
		
	# get parameters.
	if param_split is not None:
		for idx, param in enumerate(param_split):
			_, param_name, param_desc = param.split(' ', 2)
			param_name = function_comment_strip(param_name, ' *\n\t/')
			param_desc = function_comment_strip(param_desc, ' *\n\t/')
			param_split[idx] = (param_name, param_desc)
		
	# strip comments.
	brief = function_comment_strip(brief, ' *\n\t/')
	if note is not None:
		note = function_comment_strip(note, ' *\n\t/')
		
	# construct the Python function comment.
	python_fun_cmt = '"""'
	python_fun_cmt += brief + '\n'
	if param_split is not None and len(param_split) > 0:
		python_fun_cmt += '\nParameters\n----------'
		for idx, param in enumerate(param_split):
			python_fun_cmt += '\n' + param[0] + ' : ' + param_types[idx]
			python_fun_cmt += '\n\t' + param[1] + '\n'
	if note is not None:
		python_fun_cmt += '\nNote\n----\n' + note + '\n'
	python_fun_cmt += '"""'
	
	return python_fun_cmt
			
		
def function_comment_strip(comment, bad_chars):
	head_removed, tail_removed = False, False
	while not head_removed or not tail_removed:
		if comment[0] in bad_chars:
			comment = comment[1:]
			head_removed = False
		else:
			head_removed = True
		if comment[-1] in bad_chars:
			comment = comment[:-1]
			tail_removed = False
		else:
			tail_removed = True
			
	return comment

		
if __name__ == '__main__':
#  	python_code = convert_function("""
# 		if (print_to_stdout_ == 2) {
# 			std::cout << "\n===========================================================\n";
# 			std::cout << "Block gradient descent for initial median " << median_pos + 1 << " of " << medians.size() << ".\n";
# 			std::cout << "-----------------------------------------------------------\n";
# 		}
# 								""")
	
	
 	python_fun_cmt = convert_function_comment("""
	/*!
	 * @brief Returns the sum of distances.
	 * @param[in] state The state of the estimator.
	 * @return The sum of distances of the median when the estimator was in the state @p state during the last call to run().
	 */
						""", ['string', 'string'])