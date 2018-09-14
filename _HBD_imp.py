import numpy as np
from matrixofseq import mat_of_seq as mos
from autoencoder320 import ae320
import os

crd = os.path.dirname(os.path.realpath(__file__))
input_model_name = os.path.join(crd, "ecglead2generalmodel")
m1 = ae320()
m1.model = m1.load(input_model_name)


def sequence_denoising (obj):
	return


def sequence_segmentation (obj):
	seq = obj.current_dat["initial_sequence"]
	act = np.asarray(np.insert(seq, [0, -1], 0) > 1, dtype=int)
	act = np.diff(np.asarray(act[:-2] + act[1:-1] + act[2:] > 2.5, dtype=int))
	r_start = np.where(act == 1)[0]
	r_end = np.where(act == -1)[0] + 2
	peaks = [max(range(r_start[i], r_end[i]), key=lambda x: seq[x]) for i in range(len(r_start))]
	mosds = mos(seq, peaks)
	obj.current_dat["mat320"] = mosds.seq_to_mat(320)


def base_model_prediction (obj):
	dsm = obj.current_dat["mat320"]
	err = np.array([m1.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

	obj.current_dat["modl1"] = m1
	obj.current_dat["diff1"] = dsm - m1.pred(dsm)
	obj.current_dat["rank1"] = sorted(range(dsm.shape[0]), key=err.__getitem__)


def personalized_model_training (obj, normal_num=300):
	m2 = obj.current_dat["modl1"]
	normal_entry = obj.current_dat["rank1"][:normal_num]
	tsds = obj.current_dat["mat320"]
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
	obj.current_dat["modl2"] = m2
	obj.current_dat["modl1"].load(input_model_name)


def personalized_model_detection (obj):
	dsm = obj.current_dat["mat320"]
	m2 = obj.current_dat["modl2"]
	err = np.array([m2.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

	obj.current_dat["diff2"] = dsm - m2.pred(dsm)
	obj.current_dat["rank2"] = sorted(range(dsm.shape[0]), key=err.__getitem__)
