#!/usr/bin/env python

from scipy.io import whosmat, loadmat, savemat
import numpy
import re
import sys


# Adapted from http://pastebin.com/XrwTMrj5
SINGULARS= [
    (r's$', ''),
    (r'(n)ews$', '\1ews'),
    (r'([ti])a$', '\1um'),
    (r'((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\1\2sis'),
    (r'(^analy)ses$', '\1sis'),
    (r'([^f])ves$', '\1fe'),
    (r'(hive)s$', '\1'),
    (r'(tive)s$', '\1'),
    (r'([lr])ves$', '\1f'),
    (r'([^aeiouy]|qu)ies$', '\1y'),
    (r'(s)eries$', '\1eries'),
    (r'(m)ovies$', '\1ovie'),
    (r'(x|ch|ss|sh)es$', '\1'),
    (r'([m|l])ice$', '\1ouse'),
    (r'(bus)es$', '\1'),
    (r'(o)es$', '\1'),
    (r'(shoe)s$', '\1'),
    (r'(cris|ax|test)es$', '\1is'),
    (r'(octop|vir)i$', '\1us'),
    (r'(alias|status)es$', '\1'),
    (r'^(ox)en', '\1'),
    (r'(vert|ind)ices$', '\1ex'),
    (r'(matr)ices$', '\1ix'),
    (r'(quiz)zes$', '\1'),
]


def normalize(word):
	"""
	Normalizes word by converting to lower case and stripping any whitespace
	"""
	return unicode(word).lower().strip()


def strip_punctuation(word):
	"""
	Strips any characters that are not whitespace, a-z, A-Z, 0-9, /, \, and -
	"""
	return re.sub('[^\w\s\d\//\/-]', '', word)


def singularize(word):
	"""
	Adapted from http://pastebin.com/XrwTMrj5, author unknown

	Converts the word to single case according to the regex rules given in SINGULARS
	"""
	global SINGULARS

	for (pattern, replacement) in SINGULARS:
		word, n = re.subn(pattern, replacement, word)
		if n > 0:
			break

	return word

 
def build_word_index(dataset, vocab_key):
	"""
	Builds a word-to-index dict based on the given dataset

	@param dataset: The review dataset; this arg is expected to have the key 'vocab'
		defined
	@rtype: dict
	@returns: A dict, where the key is the textual word, and the value is the corresponding
		numeric index of the word in the dataset's 'vocab' array
	"""
	return {normalize(word_cell[0]): index for (index, word_cell) in enumerate(dataset[vocab_key][0])}


def find_word(word_index, word):
	"""
	Given a word, and a word index, this function will perform the
	following step in order to convert word to a numeric index:

	1. If word occurs in word_index verbatim, return the (index, word)
	2. word = normalize(word); if word occurs in word_index, return (index, word)
	3. Attempt to singularize the word; if the singularized word in word_index, return the (index, word)
	4. word = strip_punctuation(word); if word occurs in word_index, return the (index, word)

	otherwise raise a ValueError

	Returns a tuple of the form: (int: word-index, str: word) on success
	"""
	original_word = word
	verbatim = lambda word: word

	for f in (verbatim, normalize, singularize, strip_punctuation):
		word = f(word)
		if word in word_index:
			return (word_index[word], word)

	raise ValueError("'%s' / '%s' not found in word index" % (original_word, word))	


def bigrams(word_index, datasets, failed_set=None, print_messages=False):
	"""
	Returns an iterator that yields a tuple of the following form:

		(int: observation index dataset.counts, (int: bigram-1st-word index, int: bigram-2nd-word index)),

		e.g. (306, (25217, 56423))

	@type word_index: dict
	@param word_index: The word index dict created by build_word_index()
	@type datasets: list
	@param datasets:
	@type failed_set:
	@param failed_set:
	@type print_messages: bool
	@pparam print_messages: If True, output messages will be displayed for invalid
		bigrams encountered
	"""
	line = '*' * 80

	# Dataset is treated as numpy.ndarray

	for dataset in datasets:

		# These indices were determined by using ipython to interactively
		# explore the structure of the data. 
	
		for (i, row) in enumerate(dataset[0]):

			# The 5 cell contains the actual list of words. I don't know what indices 0-4
			# contain other than file format info/junk.

			words = row[5]
			N = len(words)

			for j in range(N-1):

				pair = [(None, None), (None, None)]  # Each bigram is itself a tuple of (index, word)
				failed_words = [None, None]
				skip = False

				for (k, word) in enumerate((words[j][0][0], words[j+1][0][0])):
					try:
						pair[k] = find_word(word_index, word)
					except:
						failed_words[k] = word
						skip = True

				if skip:
					if print_messages:
						print line
						print "Bad bigram -> (%s, %s) " % tuple(failed_words)
						print line

					# Keep track of failed words--optional:
					if isinstance(failed_set, set):
						for failed_word in failed_words:
							if failed_word is not None:
								failed_set.add(failed_word)

					continue

				yield (i, (pair[0][0], pair[1][0]))


def build_bigram_counts(word_index, datasets):
	"""
	@type word_index: dict
	@param word_index: The word index dict created by build_word_index()
	@type *datasets:
	@param *datasets:
	"""
	failed_set = set()
	prev_row = None
	counts = {}

	for (row, bigram) in bigrams(word_index, datasets, failed_set=failed_set):

		if bigram not in counts:
			counts[bigram] = 1
		else:
			counts[bigram] += 1

		if row != prev_row:
			print row

		prev_row = row

	return counts


def build_sparse_array(counts):
	"""
	Builds the I,J, and S arrays for use by the Matlab sparse() function

	@type counts: dict
	@param counts:
	@rtype: tuple
	@returns: 3-element tuple of numpy.ndarrays, consisting of I,J, and S
		each of length N to be used with the Matlab sparse() function 
	"""
	N = len(counts)
	dim = (N, 1)
	I = numpy.zeros(dim)
	J = numpy.zeros(dim)
	S = numpy.zeros(dim)

	for (k, (bigram, count)) in enumerate(counts.items()):
		(word_i, word_j) = bigram
		# Normalize to Marlab's 1 based indexing scheme:
		I[k], J[k], S[k] = word_i+1, word_j+1, count

	return I, J, S


def save_sparse_array(filename, I, J, S):
	"""
	Given I, J, and S arrays, this function will write them to a matlab .mat file
	"""
	savemat(filename, {'I': I, 'J': J, 'S': S})


if __name__ == "__main__":

	if len(sys.argv) < 4:
		print "\nUsage:\t %s <input-review-dataset>.mat <input-metadata>.mat <output>.mat\n" % (sys.argv[0])
		sys.exit(-1)

	review_matfile = sys.argv[1]
	metadata_matfile = sys.argv[2]
	output_filename = sys.argv[3]

	print ">> Loading review dataset file %s..." % (review_matfile)
	review_dataset = loadmat(review_matfile)

	print ">> Loading metadata file %s..." % (metadata_matfile)
	metadata = loadmat(metadata_matfile)

	print ">> Building word index..."
	word_index = build_word_index(review_dataset, 'vocab')

	print ">> Building bigram counts..."
	counts = build_bigram_counts(word_index, (metadata['train_metadata'], metadata['quiz_metadata']))

	print ">> Generating sparse array..."
	I, J, S = build_sparse_array(counts)

	print ">> Writing Matlab output to %s..." % (output_filename,)
	save_sparse_array(output_filename, I, J, S)

	print "Done!"
