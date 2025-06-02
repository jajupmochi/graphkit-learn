"""
gedlibpy.py

@Author: jajupmochi
@Date: Jun 02 2025
"""
# from gklearn.gedlib import libraries_import


def GEDEnv(env_type: str = 'attr', verbose: bool = False):
	"""
	Return the GED environment with the specified type and name.

	Parameters
	----------
	env_type: str
		The type of the GED environment. Default is 'attr'. The available types are:
		- 'attr': Attribute-based environment (with complex node and edge labels).
		- 'gxl' or 'str': GXLLabel environment (with string labels).
	"""
	if env_type == 'attr':
		if verbose:
			print('Using Attribute-based GED environment.')
		from gklearn.gedlib import libraries_import, gedlibpy_attr
		return gedlibpy_attr.GEDEnvAttr()
	elif env_type in ['gxl', 'str']:
		if verbose:
			print('Using GXLLabel GED environment.')
		from gklearn.gedlib import libraries_import, gedlibpy_gxl
		return gedlibpy_gxl.GEDEnvGXL()
	else:
		raise ValueError(
			f'Unknown GED environment type: {env_type}. '
			f'Available types are: "attr", "gxl", "str".'
		)


if __name__ == '__main__':
	# Example usage
	ged_env = GEDEnv('gxl')
	ged_env.set_edit_cost('NON_SYMBOLIC', [3, 3, 1, 3, 3, 1])
	print('Edit costs set successfully for GEDEnvGXL.')

	ged_env_attr = GEDEnv('attr')
	ged_env_attr.set_edit_cost('GEOMETRIC', [3, 3, 1, 3, 3, 1])
	print('Edit costs set successfully for GEDEnvAttr.')
