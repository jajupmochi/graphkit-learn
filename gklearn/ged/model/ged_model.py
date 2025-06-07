"""
ged_model

A wrapper for the GED model.

@Author: jajupmochi
@Date: Jun 06 2025
"""


# todo: write test code.

def GEDModel(*args, use_global_env: bool = True, **kwargs):
	"""
	ged_model

	A wrapper for the GED model.

	Currently, there are two implementations of the GED model:

	- The GEDModel class using a GEDEnv as a global environment inside the class for testing purposes.
	Compared to the previous local version, this implementation can be at least up to 25x and 5x faster,
	respectively with and without parallelization, but it is not very memory efficient.
	Check comments in `profile_ged_model.py` and `profile_ged_model_cross_matrix.py` in
	`gklearn/expeirments/ged/ged_model/` for the performance comparison.

	- The GEDModel class creating a GEDEnv locally inside the pairwise distance computation for
	each pair of graphs. This can be a bit time efficient, but also super slow.

	We have not yet optimized the automated choice of which implementation to use,
	so we leave it to the user to choose. In default, the global environment is used.

	Parameters
	----------
	args : tuple
		Positional arguments to pass to the GED model.

	use_global_env : bool, optional
		If True, use the global environment to import the GED model. Default is True.

	kwargs : dict
		Keyword arguments to pass to the GED model.

	Returns
	-------
	GEDModel
		A GED model instance.
	"""
	if use_global_env:
		from gklearn.ged.model.ged_model_global_env import GEDModel
	else:
		from gklearn.ged.model.ged_model_local_env import GEDModel
	return GEDModel(*args, **kwargs)
