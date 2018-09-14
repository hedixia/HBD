from HBD import HBD
import numpy as np
import os
filename = os.path.join("C:\D\FUSRP_RES\mitbih\MIT_BIH_dat", 'x109')
dataset = np.genfromtxt(filename, delimiter=",")
wave = dataset[0]
print(wave.shape)

hbd = HBD()
hbd.fit(wave)
d2 = hbd.current_dat["diff2"]
print(d2)
print(d2.shape)