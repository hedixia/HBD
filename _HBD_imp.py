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
	obj.current_dat["mosds"] = mosds


def base_model_prediction (obj):
	ds_seg = obj.current_dat["mat320"]
	err = np.array([m1.evaluate(ds_seg[i:i + 1]) for i in range(ds_seg.shape[0])])

	obj.current_dat["modl1"] = m1
	obj.current_dat["diff1"] = ds_seg - m1.pred(ds_seg)
	obj.current_dat["rank1"] = np.argsort(err)


def personalized_model_training (obj):
	obj.pmodle = 'pca'
	normal_entry = obj.current_dat["rank1"][:320]
	trds = obj.current_dat["mat320"][normal_entry, :]
	if obj.pmodle == 'nn':
		m2 = obj.current_dat["modl1"]
		m2.fit(trds, epochs=240, verbose=obj.verbose)
	else:
		from sklearn.decomposition import PCA

		def temp_pred (obj, ds):
			return obj.inverse_transform(obj.transform(ds))

		def temp_eval (obj, ds):
			diff = ds - temp_pred(obj, ds)
			return (diff * diff).sum(axis=1)

		PCA.pred = temp_pred
		PCA.evaluate = temp_eval

		m2 = PCA(n_components=10)
		m2.fit(trds)

	obj.current_dat["modl2"] = m2


def personalized_model_detection (obj):
	dsm = obj.current_dat["mat320"]
	m2 = obj.current_dat["modl2"]
	err = np.array([m2.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

	obj.current_dat["diff2"] = dsm - m2.pred(dsm)
	obj.current_dat["rank2"] = np.argsort(err)
	obj.current_dat["err"] = err


def detection_labeling (obj):
	mosds = obj.current_dat["mosds"]
	err = obj.current_dat["err"]
	if obj.detection_type == "direct":
		pw_label = err
	elif obj.detection_type == "probability":
		pw_label = np.exp(-err / err.mean() / 2)
	else:
		pw_label = err

	obj.label_ = mosds.mat_to_seq(pw_label)
