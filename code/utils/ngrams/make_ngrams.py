#!/usr/bin/env python

from scipy.io import whosmat, loadmat, savemat
import numpy
from itertools import chain
import re
import sys
import os


def normalize(word):
	"""
	Normalizes word by converting to lower case and stripping any whitespace
	"""
	return unicode(word).lower().strip()


def build_vocab_index(vocab, key_func=normalize):
	"""
	Builds a word-to-index dict based on the given dataset

	@param vocab: Review dataset vocab array
	@rtype: dict
	@returns: A dict, where the key is the textual word, and the value is the corresponding
		numeric index of the word in the dataset's 'vocab' array
	"""
	return {key_func(word_cell[0]): index for (index, word_cell) in enumerate(vocab)}


def normalize_words(vocab_index, words):
	"""
	Given a list of words comprising the review text, this function will
	produced a normalized list of words by converting all words to lowercase,
	splitting on punctuation, and finally only including words in the final
	output that are in vocab_index.

	@param vocab_index: The vocab index built by build_vocab_index()
	@param words: The list of words to normalize
	"""
	return [word for word in chain.from_iterable((re.split(r'[\W]', word, flags=re.U | re.I) \
		                     for word in map(lambda s: s.lower(), words))) if len(word) > 0  and word in vocab_index]


def ngrams(N, vocab_index, datasets, skipped_rows=None, print_messages=True):
	"""
	Returns an iterator that yields a tuple of the following form:

		(int: observation index dataset.counts, (int: ngram-word-0, int: ngram-word-1, ..., int: ngram-word-N-1)),

		e.g. (306, (25217, 56423, 4784))

	@param N: The n-gram size
	@param vocab_index: The vocab index dict created by build_vocab_index()
	@param datasets: A list of datasets to process the ngrams of
	@type print_messages: bool
	@param print_messages: If True, output messages will be displayed for invalid
		ngrams encountered
	"""
	# Dataset is treated as numpy.ndarray
	for dataset in datasets:

		# These indices were determined by using ipython to interactively
		# explore the structure of the data. 
		for (i, data) in enumerate(dataset[0]):

			# The 5 cell contains the actual list of words:
			words = normalize_words(vocab_index, [word[0][0] for word in data[5]])
			word_count = len(words)

			if word_count < N:
				# Keep track of skipped rows--optional:
				if isinstance(skipped_rows, list):
					skipped_rows.append((i, words))
				continue

			# For N, there are word_count - N - 1 ngrams per data row:
			for j in range(word_count - (N - 1)):
				yield (i, tuple([vocab_index[words[j + k]] for k in range(N)]))


def build_ngram_counts(N, vocab_index, datasets):
	"""
	Builds a list of ngram count dicts, where the number of returned dicts is N-1. 
	The i-th dict represents the counts for the (i-th word, i-th+1 word) in the ngram

	@param N: The n-gram size
	@param vocab_index: The word index dict created by build_vocab_index()
	@param datasets: A list of datasets to count the ngrams of
	@returns: A list of ngram count dicts, where the number of returned dicts is N-1.
	"""
	skipped_rows = []
	counts = [{} for i in range(N - 1)]
	prev_row = None

	def pairs(ngram):
		for i in range(len(ngram) - 1):
			yield (ngram[i], ngram[i+1])

	for (row, ngram) in ngrams(N, vocab_index, datasets, skipped_rows=skipped_rows):
		for (i, word_pair) in enumerate(pairs(ngram)):
			# This is done to make sure the same number of keys exist in
			# each dict in counts:
			for (j, count_dict) in enumerate(counts):
				if i == j:
					if word_pair not in count_dict:
						count_dict[word_pair] = 1
					else:
						count_dict[word_pair] += 1
				else:
					if word_pair not in count_dict:
						count_dict[word_pair] = 0

		if row != prev_row:
			print '-> build_ngram_counts: row %s done' % (row,)

		prev_row = row

	return (counts, skipped_rows)


def build_sparse_array(N, counts):
	"""
	Builds the I, J, and S arrays for use by the Matlab sparse() function

	For a vocab vector of size P and ngrams of size N, the resulting matrix will be P x P(N - 1)

	@param N: The n-gram size
	@param counts: The list of count dicts returned from build_ngram_counts.
	@rtype: tuple
	@returns: 3-element tuple of numpy.ndarrays, consisting of I,J, and S
		each of length N to be used with the Matlab sparse() function 
	"""
	P = len(counts[0])
	M = P * (N - 1)
	dim = (M, 1)
	I = numpy.zeros(dim)
	J = numpy.zeros(dim)
	S = numpy.zeros(dim)

	for (i, count_dict) in enumerate(counts):
		for (j, (word_pair, count)) in enumerate(count_dict.items()):
			# Normalize to Matlab's 1 based indexing scheme:
			k = (i * P) + j
			I[k], J[k], S[k] = (word_pair[0] + 1), (word_pair[1] + 1) * (i + 1), count

	return I, J, S


def print_metrics(counts, vocab, skipped_rows):
	"""
	Prints frequency metrics for the given counts
	"""
	master_counts = {}
	for count_dict in counts:
		master_counts.update(count_dict)

	# Sort by counts in descending order:
	sorted_tuples = sorted(master_counts.items(), lambda x, y: -cmp(x[1], y[1]))

	# Frequencies
	with open(os.getcwd() + "/report_ngram_freq.txt", 'w') as f:
		for (word_pair, count) in sorted_tuples:
			try:
				print >>f, "%s %s | %s %s | %s" % \
					(word_pair[0] + 1,  # Normalize to Matlab's 1 based indexing scheme:
					 word_pair[1] + 1,  # Normalize to Matlab's 1 based indexing scheme:
					 vocab[word_pair[0]][0].encode('utf-8'), 
					 vocab[word_pair[1]][0].encode('utf-8'), 
					 count)
			except:
				raise

	# Skipped rows:
	with open(os.getcwd() + "/report_skipped_rows.txt", 'w') as f:
		for (row, words) in skipped_rows:
			try:
				print >>f, "%s %s" % (row, words.encode('utf-8'))
			except:
				raise


def save_sparse_array(filename, I, J, S):
	"""
	Given I, J, and S arrays, this function will write them to a matlab .mat file
	"""
	savemat(filename, {'I': I, 'J': J, 'S': S})


if __name__ == "__main__":

	if len(sys.argv) < 5:
		print "\nUsage:\t %s <N> <input-review-dataset>.mat <input-metadata>.mat <output>.mat\n" % (sys.argv[0])
		sys.exit(-1)

	N = int(sys.argv[1])
	review_matfile = sys.argv[2]
	metadata_matfile = sys.argv[3]
	output_filename = sys.argv[4]

	print ">> Loading review dataset file %s..." % (review_matfile)
	review_dataset = loadmat(review_matfile)

	print ">> Loading metadata file %s..." % (metadata_matfile)
	metadata = loadmat(metadata_matfile)

	vocab = review_dataset['vocab'][0]

	print ">> Building word index..."
	vocab_index = build_vocab_index(vocab)

	print ">> Building ngram (N = %s) counts..." % (N,)
	(counts, skipped_rows) = build_ngram_counts(N, vocab_index, (metadata['train_metadata'], metadata['quiz_metadata']))

	print ">> Generating sparse array..."
	I, J, S = build_sparse_array(N, counts)

	print ">> Writing Matlab output to %s..." % (output_filename,)
	save_sparse_array(output_filename, I, J, S)

	print ">> Printing ngram frequencies..."
	print_metrics(counts, vocab, skipped_rows)

	print "Done!"
