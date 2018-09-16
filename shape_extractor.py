import numpy as np


class Shape_Extractor:
	def __init__ (self, seq, intrim_cutoff):
		self.seq = np.asarray(seq)
		self.n_point = len(seq)
		self.cutoff = self._cutoff_input(intrim_cutoff)
		self.n_seg = len(self.cutoff)
		self.length_list = self._get_length_list(self.cutoff, self.n_point)
		self.lseq = [self.seq[self.cutoff[i]: self.cutoff[i] + self.length_list[i]] for i in range(self.n_seg)]

	def seq_to_mat (self, output_len):
		length_list = self.length_list
		mat = self.lseq

		def tempf (i, j):
			x = j / (output_len - 1) * (length_list[i] - 1)
			xp = np.arange(0, length_list[i], 1)
			fp = mat[i]
			return np.interp(x, xp, fp)

		to_matrix = np.fromfunction(np.vectorize(tempf), (self.n_seg, output_len), dtype=int)
		return np.asarray(to_matrix)

	def get_seq (self, from_, to_, seq=None):
		if seq is None:
			seq = self.seq
		try:
			return seq[self.cutoff[from_]: self.cutoff[to_ + 1] + 1]
		except IndexError:
			return seq[self.cutoff[from_]:]

	def mat_to_seq (self, input_matrix):
		input_matrix = np.asarray(input_matrix)
		assert input_matrix.shape[0] == self.n_seg
		if input_matrix.ndim is 1:
			input_matrix = np.repeat(input_matrix, 2).reshape(-1, 2)
		input_len = input_matrix.shape[1]
		length_list = self.length_list

		def tempf (i, j):
			x = j / (length_list[i] - 1) * (input_len - 1)
			xp = np.arange(0, input_len, 1)
			fp = input_matrix[i]
			return np.interp(x, xp, fp)

		lseq = [[tempf(i, j) for j in range(self.length_list[i])] for i in range(self.n_seg)]
		return self._join_lseq(lseq)

	@staticmethod
	def _cutoff_input (intrim_cutoff):
		set_cutoff = set(intrim_cutoff)
		set_cutoff.add(0)
		cutoff = sorted(set_cutoff)
		return cutoff

	@staticmethod
	def _get_length_list (cutoff, n):
		if cutoff[-1] != n - 1:
			cutoff.append(n - 1)
		length_list = np.diff(cutoff)
		length_list += 1
		length_list.tolist()
		return length_list

	@staticmethod
	def _join_lseq (lseq):
		for i in range(len(lseq) - 1):
			ave = (lseq[i][-1] + lseq[i + 1][0]) / 2
			lseq[i] = lseq[i][:-1]
			lseq[i + 1][0] = ave
		lseq = np.concatenate(lseq)
		return lseq
