import time
time_list = [time.time()]
from HBD import HBD
import numpy as np
import os

time_list.append(time.time())
crd = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(crd, 'x109')
dataset = np.genfromtxt(filename, delimiter=",")
wave = dataset[0]

time_list.append(time.time())

hbd = HBD(verbose=True)
hbd.fit(wave)
time_list.append(time.time())

time_list = np.asarray(time_list) - time_list[0]
print(time_list)
print(np.diff(time_list))