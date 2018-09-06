import numpy as np
from matrixofseq import mat_of_seq as mos
from autoencoder320 import ae320
import os

crd = os.path.dirname(os.path.realpath(__file__))


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

	def __init__ (self, s):
		self.s = s
		self.current_dat = {"initial_sequence": np.asarray(s)}
		self.progress_check = {}

	def predict (self):
		self.denoising_process()
		self.differencing_process()
		self.retraining_process()
		self.detection_process()

	@_run_once_for_each_object
	def denoising_process (self):
		self.chopping()

	@_run_once_for_each_object
	def differencing_process (self):
		self.manifold1_diff()

	@_run_once_for_each_object
	def retraining_process (self):
		self.manifold2_training()
		self.manifold2_diff()

	@_run_once_for_each_object
	def detection_process (self):
		return

	def chopping (self):
		seq = self.current_dat["initial_sequence"]
		act = np.asarray(np.insert(seq, [0, -1], 0) > 1, dtype=int)
		act = np.diff(np.asarray(act[:-2] + act[1:-1] + act[2:] > 2.5, dtype=int))
		r_start = np.where(act == 1)[0]
		r_end = np.where(act == -1)[0] + 2
		peaks = [max(range(r_start[i], r_end[i]), key=lambda x: seq[x]) for i in range(len(r_start))]
		mosds = mos(seq, peaks)
		self.current_dat["mat320"] = mosds.seq_to_mat(320)

	def manifold1_diff (self):
		m1 = self._m1_load()
		dsm = self.current_dat["mat320"]
		err = np.array([m1.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

		self.current_dat["modl1"] = m1
		self.current_dat["diff1"] = dsm - m1.pred(dsm)
		self.current_dat["rank1"] = sorted(range(dsm.shape[0]), key=err.__getitem__)

	def manifold2_training (self, normal_num=300):
		m2 = self.current_dat["modl1"]
		normal_entry = self.current_dat["rank1"][:normal_num]
		tsds = self.current_dat["mat320"]
		trds = tsds[normal_entry, :]
		ptserr = 1
		for j in range(3):
			print("iteration ", j)
			trerr = m2.fit(trds)
			print("training error: ", trerr)
			tserr = m2.evaluate(tsds)
			print("test error: ", tserr)
			if ptserr < tserr:
				break
			else:
				ptserr = tserr
		self.current_dat["modl2"] = m2
		self._m1_load()

	def manifold2_diff (self):
		dsm = self.current_dat["mat320"]
		m2 = self.current_dat["modl2"]
		err = np.array([m2.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

		self.current_dat["diff2"] = dsm - m2.pred(dsm)
		self.current_dat["rank2"] = sorted(range(dsm.shape[0]), key=err.__getitem__)

	def _m1_load (self, modelname="ecglead2generalmodel"):
		m1 = ae320()
		m1.model = m1.load(os.path.join(crd, modelname))
		self.current_dat["modl1"] = m1
		return m1


x = HBD([1])
x.predict()
print(x.progress_check)
print(x.__dict__)
