"""
bind_errors



@Author: jajupmochi
@Date: May 31 2025
"""


#####################
##ERRORS MANAGEMENT##
#####################

class Error(Exception):
	"""
		Class for error's management. This one is general.
	"""
	pass


class EditCostError(Error):
	"""
		Class for Edit Cost Error. Raise an error if an edit cost function doesn't exist in the library (not in list_of_edit_cost_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""


	def __init__(self, message):
		"""
			Inits the error with its message.

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message


class MethodError(Error):
	"""
		Class for Method Error. Raise an error if a computation method doesn't exist in the library (not in list_of_method_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""


	def __init__(self, message):
		"""
			Inits the error with its message.

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message


class InitError(Error):
	"""
		Class for Init Error. Raise an error if an init option doesn't exist in the library (not in list_of_init_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""


	def __init__(self, message):
		"""
			Inits the error with its message.

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message
