"""
Classes for loading data into memory.
"""

import numpy as np
import os, sys

class DataLoader(object):
	"""Contains functionality to convert tokens to integers."""

	def __init__(self, data_dir, token='char', encoding='utf-8', split='train', convert=True):
		"""Standard __init__ method."""
		input_path = os.path.join(data_dir,split + '.txt')
		self.convert = convert
		
        print("Reading text file")
		with open(input_path, 'r', encoding=encoding) as f:
			text = f.read()

		# Convert text to list
        if token == 'char'
		    self.text = list(text)
        else if token == 'word'
            self.text = text.split()

		# Vocabulary file needs to be explicitly provided in data_dir
		# Reuse old vocabulary file
		print("Reading pre-processed vocabulary")
		saved_vocab = os.path.join(data_dir, "vocab")
		with open(saved_vocab, 'r', encoding=encoding) as f:
			self.rev_vocab = f.read().split('\n')

		self.vocab = {word: i for i, word in enumerate(self.rev_vocab)}

        # At test time, return the input list of tokens and vocabulary
        # This allows OoV tokens to be handled as per requirement
		if not self.convert:
			return
		else:
			# Convert the tokens to integers based on vocabulary map
			self.data = np.array(list(map(self.vocab.get, self.text)))

class BatchLoader(object):
	"""This class is used to build batches of data."""

	def __init__(self, data_loader, batch_size, timesteps):
		"""Standard __init__ function."""
		# Copying pointers
		self.data_loader = data_loader
		self.vocab = data_loader.vocab
		if data_loader.convert:
			data = data_loader.data
		else:
			data = data_loader.text

		self.batch_size = batch_size
		self.timesteps = timesteps

		self.num_batches = num_batches = int(len(data) / (batch_size * timesteps))
		# When the data (tensor) is too small
		if num_batches == 0:
			print("Not enough data. Make seq_length and batch_size small.")
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

	def next_batch(self):
		"""Output the next batch."""
		x = self.x_batches[self.pointer]
		y = self.y_batches[self.pointer]

		self.pointer += 1
		return x, y
	
	def reset_batch_pointer(self):
		self.pointer = 0
