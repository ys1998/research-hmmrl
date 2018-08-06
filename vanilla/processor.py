"""This file loads text, processes it and converts it into batches."""
from strings import FILES, LOGS, ERRORS

from six.moves import cPickle
import codecs
import numpy as np
import os
import sys
import multiprocessing as mp
from multiprocessing import sharedctypes
import time


class DataLoader(object):
	"""Contains functionality to convert text to integers."""

	def __init__(self, args):
		"""Standard __init__ method."""
		self.args = args
		input_path = os.path.join(args.data_dir,"train.txt")
		print(LOGS[0])
		# codecs is needed due to utf-8 setting
		# This is essential for non-ASCII characters
		with open(input_path, "r") as f:
			text = f.read()

		# Convert newlines to spaces
		# Then convert to characters
		self.text = list(text.replace('\n', ' '))

		# vocab_file needs to be explicitly provided in data_dir
		# Re-Use old vocabulary file
		print(LOGS[1])
		saved_vocab = os.path.join(args.data_dir, args.vocab)
		with open(saved_vocab, 'r') as f:
			self.rev_vocab = f.read().split('\n')

		self.vocab = {word: i for i, word in enumerate(self.rev_vocab)}

		args.vocab_size = self.vocab_size = len(self.vocab)

		if args.mode == 'test':
			return

		# Convert the text tokens to integers based on vocab map
		self.data = np.array(list(map(self.vocab.get, self.text)))


class BatchLoader(object):
	"""This class is used to build batches of data."""

	def __init__(self, args, data_loader):
		"""Standard __init__ function."""
		# Copying pointers
		self.args = args
		self.data_loader = data_loader
		self.vocab = data_loader.vocab
		self.vocab_size = data_loader.vocab_size
		data = data_loader.data
		self.batch_size = batch_size = args.config.batch_size
		self.timesteps = timesteps = args.config.timesteps

		self.num_batches = num_batches = int(len(data) / (batch_size * timesteps))
		# When the data (tensor) is too small
		if num_batches == 0:
			print(ERRORS[0])
			sys.exit()

		xdata = data[:num_batches * batch_size * timesteps]
		ydata = np.copy(xdata)

		# Building output tokens - next token predictors
		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]

		self.xdata = xdata
		self.ydata = ydata

		# Splitting x, y and constants into batches.
		# Frequency batches are not generated now to save memory.
		self.x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
		self.y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)

		self.pointer = 0

	def next_batch(self, l1):
		"""Output the next batch and corresponding frequencies."""
		x = self.x_batches[self.pointer]
		y = self.y_batches[self.pointer]

		self.pointer += 1
		return x, y
	
	def reset_batch_pointer(self):
		self.pointer = 0

def eval_loader(args, vocab, split):
	"""Convert raw evaluation data to correct format."""
	filename = "%s.txt" % (split)
	input_path = os.path.join(args.data_dir, filename)
	with codecs.open(input_path, 'r', encoding='utf-8') as f:
		text = f.read()
	timesteps = args.config.timesteps
	# Fix data to add <s> and </s> characters
	tokens = list(text.replace('\n', ' '))
	# Replacing all OOV with <unk>, converting to integers
	x = [vocab[c] for c in tokens]
	
	total_len = len(x)
	# pad ipa_x so the batch_size divides it exactly
	while len(x) % timesteps != 1:
		x.append(vocab[' '])
	y = np.array(x[1:]).reshape((-1, timesteps))
	x = np.array(x[:-1]).reshape((-1, timesteps))
	return x, y, total_len
