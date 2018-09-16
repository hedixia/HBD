import os

import numpy as np

from autoencoder320 import ae320
from shape_extractor import Shape_Extractor as SE

crd = os.path.dirname(os.path.realpath(__file__))
input_model_name = os.path.join(crd, "ecglead2generalmodel")
model_1 = ae320()
model_1.model = model_1.load(input_model_name)


def sequence_denoising (obj):
	return


def sequence_segmentation (obj):
	seq = obj.current_dat["initial_sequence"]
	act = np.asarray(np.insert(seq, [0, -1], 0) > 1, dtype=int)
	act = np.diff(np.asarray(act[:-2] + act[1:-1] + act[2:] > 2.5, dtype=int))
	r_start = np.where(act == 1)[0]
	r_end = np.where(act == -1)[0] + 2
	peaks = [max(range(r_start[i], r_end[i]), key=lambda x: seq[x]) for i in range(len(r_start))]
	shape_extractor_ = SE(seq, peaks)
	obj.current_dat["shape_320"] = shape_extractor_.seq_to_mat(320)
	obj.current_dat["shape_extractor_"] = shape_extractor_


def base_model_prediction (obj):
	ds_seg = obj.current_dat["shape_320"]
	err = np.array([model_1.evaluate(ds_seg[i:i + 1]) for i in range(ds_seg.shape[0])])

	obj.current_dat["model_1"] = model_1
	obj.current_dat["diff_1"] = ds_seg - model_1.pred(ds_seg)
	obj.current_dat["rank_1"] = np.argsort(err)


def personalized_model_training (obj):
	normal_entry = obj.current_dat["rank_1"][:320]
	trds = obj.current_dat["shape_320"][normal_entry, :]
	if obj.model_type == 'nn':
		model_2 = obj.current_dat["model_1"]
		model_2.fit(trds, epochs=240, verbose=obj.verbose)
	else:
		from sklearn.decomposition import PCA

		def temp_pred (obj, ds):
			return obj.inverse_transform(obj.transform(ds))

		def temp_eval (obj, ds):
			diff = ds - temp_pred(obj, ds)
			return (diff * diff).sum(axis=1)

		PCA.pred = temp_pred
		PCA.evaluate = temp_eval

		model_2 = PCA(n_components=10)
		model_2.fit(trds)

	obj.current_dat["model_2"] = model_2


def personalized_model_detection (obj):
	dsm = obj.current_dat["shape_320"]
	model_2 = obj.current_dat["model_2"]
	err = np.array([model_2.evaluate(dsm[i:i + 1]) for i in range(dsm.shape[0])])

	obj.current_dat["diff_2"] = dsm - model_2.pred(dsm)
	obj.current_dat["rank_2"] = np.argsort(err)
	obj.current_dat["err"] = err


def detection_labeling (obj):
	mosds = obj.current_dat["mosds"]
	err = obj.current_dat["err"]
	if obj.detection_type == "direct":
		pw_label = err
	elif obj.detection_type == "prob":
		pw_label = np.exp(-err / err.mean() / 2)
	else:
		pw_label = err

	obj.label_ = mosds.mat_to_seq(pw_label)
