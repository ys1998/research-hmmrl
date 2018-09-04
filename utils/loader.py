"""
Classes for loading data into memory.
"""

import numpy as np
import os, sys

class DataLoader(object):
	"""Contains functionality to convert tokens to integers."""

	def __init__(self, data_dir, mode='train', tokenize_func=None, encode_func=None):
		"""Standard __init__ method."""
		self.tokenize_func = tokenize_func
		self.encode_func = encode_func
		self.mode = mode

		if mode == 'train':
			input_path = os.path.join(data_dir,'train.txt')
			print("Reading %s" % input_path)
			with open(input_path, 'r', encoding='utf-8') as f:
				train_text = f.read()

			input_path = os.path.join(data_dir,'valid.txt')
			print("Reading %s" % input_path)
			with open(input_path, 'r', encoding='utf-8') as f:
				val_text = f.read()

			# Convert text to tokens
			tokens, vocabs = tokenize_func(train_data=train_text, val_data=val_text, save_dir=data_dir)
			# Store data and vocabular[y|ies]
			self.data = tokens
			self.vocabs = vocabs

		elif mode == 'test':
			input_path = os.path.join(data_dir,'test.txt')
			print("Reading %s" % input_path)
			with open(input_path, 'r', encoding='utf-8') as f:
				test_text = f.read()
			
			# Convert text to tokens
			tokens, _ = tokenize_func(test_data=test_text)
			# Store data
			self.data = tokens

class BatchLoader(object):
	"""This class is used to build batches of data."""

	def __init__(self, data_loader, batch_size, timesteps, mode='train'):
		"""Standard __init__ function."""
		# Copying pointers
		self.data_loader = data_loader
		self.batch_size = batch_size
		self.timesteps = timesteps

		if mode == 'train':
			self.data = data_loader.data['train']
		elif mode == 'val':
			self.data = data_loader.data['val']
		elif mode == 'test':
			self.data = data_loader.data['test']

		self.num_batches = num_batches = int(len(self.data) / (batch_size * timesteps))
		# When the data (tensor) is too small
		if num_batches == 0:
			print("Not enough data. Make seq_length and batch_size small.")
			sys.exit()

		xdata = self.data[:num_batches * batch_size * timesteps]
		ydata = np.copy(xdata)

		# Building output tokens - next token predictors
		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]

		# Encode each token as a `dict` of indices in multiple embeddings
		#
		# Eg. 'hello' = {0: [67], 1: [8,5,12,12,15], 2: [83]}
		# 0 - word space
		# 1 - char space
		# 2 - morpheme space
		#
		self.xdata = np.asarray(data_loader.encode_func(xdata, data_loader.vocabs))
		self.ydata = np.asarray(data_loader.encode_func(ydata, data_loader.vocabs))

		# Splitting data into batches.
		self.x_batches = np.split(self.xdata.reshape(batch_size, -1), num_batches, 1)
		self.y_batches = np.split(self.ydata.reshape(batch_size, -1), num_batches, 1)

		self.pointer = 0

	def next_batch(self):
		"""Output the next batch."""
		x = self.x_batches[self.pointer]
		y = self.y_batches[self.pointer]

		self.pointer += 1
		return x, y
	
	def reset_batch_pointer(self):
		self.pointer = 0
