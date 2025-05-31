"""
gedlibpy

Helper for GEDLIBPY Python bindings.

@Author: jajupmochi
@Date: May 31 2025
"""
from gedlibpy_gxl import GEDEnvGXL
from gedlibpy_attr import GEDEnvAttr


def GEDEnv(env_type='attr'):
    """
    A factory function that returns the appropriate GED environment.

    This function provides a unified interface for different GED environments, such as GXL
    and attribute-based graphs.
    The specific environment will be selected based on the provided parameter `env_type`.

    Parameters
    ----------
    env_type : str
        The type of the GED environment to initialize. Options are 'gxl', 'str', or 'attr'.
        'gxl' and 'str' initialize a GXL-based environment, where node and edge labels are strings.
        The default is 'attr', which initializes an attribute-based environment,
        where node and edge labels can be dictionaries with string keys and values of
        various types (namely, string, int, float, list/np.array of strings,
        and list/np.array of integers, and list/np.array of floats).

    Returns
    -------
    ged_env : GEDEnvGXL or GEDEnvAttr
        An instance of the appropriate GED environment class based on the specified type.

    Raises
    ------
    ValueError
        If the provided `env_type` is not recognized (not 'gxl', 'str', or 'attr').
    """
    env_type = env_type.lower()
    if env_type in ['gxl', 'str']:
        return GEDEnvGXL()
    elif env_type == 'attr':
        return GEDEnvAttr()
    else:
        raise ValueError(
            f'Unknown environment type: {env_type}. Please use "gxl", "str", or "attr".'
            f'"gxl" and "str" are equivalent and use string labels for nodes and edges, '
            f'while "attr" uses attribute-based labels with dictionaries with string '
            f'keys and values of various types, namely, string, int, float, '
            f'list/np.array of strings, list/np.array of integers, and list/np.array of floats.'
        )
