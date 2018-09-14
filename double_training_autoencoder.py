# Imports
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from matrixofseq import mat_of_seq as mos

crd = os.path.dirname(os.path.realpath(__file__))
viewseq = [2305, 2300, 2248, 2000]

# Model
from autoencoder320 import aenn as a320


def plot (i, mosds, predseq, err, SOI, vlen=1080, text=""):
	if isinstance(i, int):
		selflen = mosds.length_list[i]
		start_ = mosds.cutoff[i] - (vlen - selflen) // 2
		end_ = start_ + vlen
		plt.figure()
		fig = plt.subplot()
		fig.plot(mosds.seq[start_: end_])
		fig.plot(predseq[start_: end_])
		fig.plot(mosds.seq[start_: end_] - predseq[start_: end_])
		fig.axvspan(mosds.cutoff[i] - start_, mosds.cutoff[i + 1] - start_, alpha=0.5, color='yellow')
		plt.title(text + " " + str(i))
		plt.text(800, -2.5, "SOI=" + "{:10.4f}".format(SOI[i]))
		plt.ylim(-3, 3)
		print(i, err[i], SOI[i])
		plt.savefig(text + "_" + str(i) + '.png')
	else:
		for j in i:
			plot(j, mosds, predseq, err, SOI, vlen=vlen, text=text)


def dataoutput (dsname, foldername=None, dataset=None, **kwargs):
	import csv
	dsname = str(dsname)
	for key in kwargs:
		dataoutput(dsname, key, kwargs[key])

	if foldername is None:
		return

	try:
		os.mkdir(os.path.join(crd, foldername))
	except FileExistsError:
		pass
	dataset = np.asarray(dataset)
	if dataset.ndim == 1:
		dataset = dataset.reshape(-1, 1)
	with open(os.path.join(crd, foldername, dsname), "w", newline="") as csvwfile:
		csvwriter = csv.writer(csvwfile)
		csvwriter.writerows(dataset)
	print(dsname + " " + foldername + " finished.")


# Item
item = 109
indatname = "x" + str(item)
base_model = "ecglead2generalmodel"

a320.model = a320.load(os.path.join(crd, base_model))

# Data
filename = os.path.join("C:\D\FUSRP_RES\mitbih\MIT_BIH_dat", indatname)
dataset = np.genfromtxt(filename, delimiter=",")
wave = dataset[0]
print(wave.shape)

# Slices
seq = wave
act = np.asarray(np.insert(seq, [0, -1], 0) > 1, dtype=int)
act = np.diff(np.asarray(act[:-2] + act[1:-1] + act[2:] > 2.5, dtype=int))
r_start = np.where(act == 1)[0]
r_end = np.where(act == -1)[0] + 2
peaks = [max(range(r_start[i], r_end[i]), key=lambda x: seq[x]) for i in range(len(r_start))]
mosds = mos(wave, peaks)
dataset = mosds.seq_to_mat(320)
print(dataset.shape)

# First Layer Outlier Ranking
normal_fetch = 300
tsds = dataset[:, :]
err = np.array([a320.evaluate(tsds[i:i + 1]) for i in range(tsds.shape[0])])
SOI = err / err.mean()
print(sorted(SOI, reverse=True)[:5])
print("test error: ", np.mean(err))
normalentry = np.argpartition(err, normal_fetch)[:normal_fetch]

# Plotting
pred_mat = a320.pred(tsds)
pred_seq = mosds.mat_to_seq(pred_mat)
plot(viewseq, mosds, pred_seq, err, SOI, text="M1")

# Retraining
trds = dataset[normalentry, :]
tsds = dataset[:, :]
print(trds.shape)
print(tsds.shape)

# Training
ptserr = 1
tserr = 1
for j in range(3):
	print("iteration ", j)
	trerr = a320.fit(trds)
	print("training error: ", trerr)
	tserr = a320.evaluate(tsds)
	print("test error: ", tserr)
	if ptserr < tserr:
		break
	else:
		ptserr = tserr

# Outliers
err = np.array([a320.evaluate(tsds[i:i + 1]) for i in range(tsds.shape[0])])
SOI = err / err.mean()
print(sorted(SOI, reverse=True)[:5])
sigentry = np.argpartition(err, -5)[-5:]
sigentry = sigentry[np.argsort(-(err[sigentry]))]
print("first 5 significant entry: ", sigentry)
err_seq = mosds.mat_to_seq(err)
dataoutput(item, err=err, err_seq=err_seq)

# Plot
pred_mat = a320.pred(tsds)
pred_seq = mosds.mat_to_seq(pred_mat)
plot(viewseq, mosds, pred_seq, err, SOI, text="M2")


plt.figure()
plt.hist(SOI, bins=1000)
plt.xlabel("SOI")
plt.ylabel("Occurrance")
plt.ylim(0, 12)
plt.xlim(0, 20)
plt.show()

diff_seq = seq - pred_seq
dataoutput("109.csv", diff_seq=diff_seq)

#CPD = rpt.Pelt(model='rbf', min_size=360)
#CPD.fit(diff_seq)
#cpd_on_diff = CPD.predict(pen=35)
