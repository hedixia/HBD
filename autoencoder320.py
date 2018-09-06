# Imports
import tensorflow as tf
from ModelGenerator import ModelGenerator


# Parameters
class Param:
	# Network Parameters
	input_size = [320]
	filters = 6
	kernel_size = (5, 1)
	pool_size = (4, 1)
	strides = (4, 1)
	padding = 'SAME'
	activation = tf.keras.layers.LeakyReLU
	activation_param = {"alpha": 0.1}
	units = 1


# Notion
C = "conv"
M = "maxpool"
U = "upsample"
F = "final"

aenn = ModelGenerator(Param.__dict__)
aenn.add_layer(C, M, C, M, C, M, syntax="E")  # Encoder
aenn.add_layer(C, U, U, U, F, syntax="D")  # Decoder
open("model_summary.txt", "w").write(aenn.summary)

class ae320 (ModelGenerator):
	def __init__(self):
		super().__init__(Param.__dict__)
		self.add_layer(C, M, C, M, C, M, syntax="E")  # Encoder
		self.add_layer(C, U, U, U, F, syntax="D")  # Decoder