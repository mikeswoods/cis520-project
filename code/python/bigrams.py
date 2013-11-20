from scipy.io import whosmat, loadmat, savemat
from numpy import ndarray, uint8


def build_bigrams(metadata, name):
	"""
	I will describe this later
	"""
	# Dataset is treated as numpy.ndarray
	dataset = metadata[name]

	# These indices were determined by using ipython to interactively
	# explore the structure of the data. 

	N = len(dataset[0])

	for (i, row) in enumerate(dataset[0]):
		prev_word = None
		# The 5 cell contains the actual list of words. I don't know what indices 1-4
		# contain other than file format info/junk.
		for (j, word_cell) in enumerate(row[5]):
			word = word_cell[0][0]
			print word, 
 

if __name__ == "__main__":
	pass