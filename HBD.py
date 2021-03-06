from _HBD_imp import *


def _run_once_for_each_object (func):
	"""
	Decorator for class attributes.
	Allows function with this decorator to run only for the first time for each object created.
	Decorated attributes turn to Null function after first called.
	These functions can be re-enabled through obj.progress_check
	"""

	def wrapper (obj, *args, **kwargs):
		if not hasattr(obj, "progress_check"):
			obj.progress_check = {}
		if obj.progress_check.get(func.__name__, 0) != 1:
			temp = func(obj, *args, **kwargs)
			obj.progress_check[func.__name__] = 1
			return temp

	return wrapper


class HBD(object):

	def __init__ (self, verbose=False, model_type='pca', detection_type="SSE", *kwargs):
		"""
		Create a HBD object
		:param verbose: Show processes if Neural Network is used
		:param model_type: Model for Personalized Training Process, default to pca
		:param detection_type: Output of detection. Default to error sums of squares (SSE), choice includes diff, prob
		:param kwargs:
		"""
		# input
		self.verbose = verbose
		self.model_type = model_type
		self.detection_type = detection_type
		self.__dict__.update(kwargs)

		# self initialization
		self.s = None
		self.current_dat = None
		self.progress_check = {}
		self.label_ = None

	def fit (self, s):
		self.s = s
		self.current_dat = {"initial_sequence": np.asarray(s)}

		self.segmentation_process()
		self.personalized_training_process()
		self.detection_process()

	def predict (self, model_type=None):
		if model_type is None:
			return self.label_
		else:
			self.model_type = model_type
			detection_labeling(self)
			return self.label_

	@_run_once_for_each_object
	def segmentation_process (self):
		sequence_denoising(self)
		sequence_segmentation(self)

	@_run_once_for_each_object
	def personalized_training_process (self):
		base_model_prediction(self)
		personalized_model_training(self)
		personalized_model_detection(self)

	@_run_once_for_each_object
	def detection_process (self):
		detection_labeling(self)
