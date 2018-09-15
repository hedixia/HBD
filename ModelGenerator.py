import numpy as np
import tensorflow as tf


class ModelGenerator:
	pp_dict = {  # str -> coefficients
		"conv": ("filters", "kernel_size", "padding"),
		"maxpool": ("pool_size", "strides"),
		"upsample": ("filters", "kernel_size", "padding", "strides"),
		"dense": ("units",),
	}

	layer_dict = {  # str -> tf.keras.layer
		"conv": tf.keras.layers.Conv2D,
		"maxpool": tf.keras.layers.MaxPool2D,
		"upsample": tf.keras.layers.Conv2DTranspose,
		"final": tf.keras.layers.Dense,
	}

	activation_set = {  # Layers followed with activation
		"conv",
		"dense",
	}

	def __init__ (self, pdict):
		self.layer_list = []  # List of non-activation layers
		self.name_list = []  # List of names of non-activation layers
		self.param_dict = dict((ltype, dict((k, pdict[k]) for k in ModelGenerator.pp_dict[ltype])) for ltype in ModelGenerator.pp_dict)
		self.param_dict["final"] = {"units": 1}
		self.pdict = pdict
		self.input_size = list(pdict["input_size"])
		self.formatted_input_size = self.input_size[:3] + [1] * (3 - len(self.input_size))
		self.model = tf.keras.models.Sequential()
		self.compiled = False
		self.summary_ = ""

	def add_layer (self, layer_type, *args, syntax=""):
		name = "_".join([syntax, str(len(self)), layer_type])
		self.name_list.append(name)
		new_layer = self.layer_dict[layer_type](name=name, **self.param_dict[layer_type])
		self.layer_list.append(new_layer)
		self.model.add(new_layer)
		if layer_type in self.activation_set:
			self.model.add(self.pdict["activation"](**self.pdict["activation_param"]))

		for i in args:
			self.add_layer(i, syntax=syntax)

	def fit (self, tr_set, epochs=5, verbose=False, loss='MSE'):
		if not self.compiled:
			self.model.compile(optimizer='adam', loss=loss)
			self.compiled = True
		tr_set = tr_set.reshape(-1, *self.formatted_input_size)
		self.model.fit(tr_set, tr_set, epochs=epochs, verbose=verbose)
		error_rate = self.model.evaluate(tr_set, tr_set, verbose=verbose)
		if verbose:
			print("error_rate = ", error_rate)
		return error_rate

	def pred (self, ts_set):
		ts_set = ts_set.reshape(-1, *self.formatted_input_size)
		xpred = self.model.predict(ts_set)
		return xpred.reshape(-1, *self.input_size)

	def evaluate (self, ts_set, verbose=False):
		ts_set = ts_set.reshape(-1, *self.formatted_input_size)
		error_rate = self.model.evaluate(ts_set, ts_set, verbose=verbose)
		if verbose:
			print("error_rate = ", error_rate)
		return error_rate

	def save (self, filepath):
		tf.keras.models.save_model(self.model, filepath=filepath)

	def load (self, filepath):
		try:
			self.model = tf.keras.models.load_model(filepath=filepath)
		except OSError:
			print("Model File Not Exist")
		return self.model

	@property
	def summary (self):
		if self.summary_ == "":
			self.fit(np.zeros([1] + self.formatted_input_size))
			self.model.summary(print_fn=self._add_summary)
		return self.summary_

	def _add_summary (self, x):
		self.summary_ += x + "\n"

	def __len__ (self):
		return len(self.layer_list)

	def __repr__ (self):
		return self.name_list.__repr__()
