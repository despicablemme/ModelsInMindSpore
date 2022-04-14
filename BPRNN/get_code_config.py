import numpy as np


def load_code_message(H_filename, G_filename):
	# parity-check matrix; Tanner graph parameters
	with open(H_filename) as f:
		# get n and m (n-k) from first line
		n, m = [int(s) for s in f.readline().split(' ')]
		k = n-m

		var_degrees = np.zeros(n).astype(np.int)  # degree of each variable node

		# initialize H
		H = np.zeros([m, n]).astype(np.int)
		f.readline()  # ignore 3 lines
		f.readline()
		f.readline()

		# create H, sparse version of H, and edge index matrices
		# (edge index matrices used to calculate source and destination nodes during belief propagation)
		var_edges = [[] for _ in range(0,n)]
		for i in range(0, n):
			row_string = f.readline().split(' ')
			var_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			var_degrees[i] = len(var_edges[i])
			H[var_edges[i], i] = 1

		# num_edges = H.sum()

	G = np.loadtxt(G_filename).astype(np.int).transpose()

	return H, G, n, m, k
