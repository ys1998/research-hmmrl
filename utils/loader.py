"""
Classes for loading data into memory.
"""

import numpy as np
import os, sys

class DataLoader(object):
	"""Contains functionality to convert tokens to integers."""

	def __init__(self, data_dir, mode='train', tokenize_func=None, encode_func=None, word_markers=True, max_word_length=65):
		"""Standard __init__ method."""
		self.tokenize_func = tokenize_func
		self.encode_func = encode_func
		self.mode = mode
		self.word_markers = word_markers

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
			tokens, vocabs = tokenize_func(
				train_data=train_text, 
				val_data=val_text, 
				save_dir=data_dir, 
				word_markers=self.word_markers,
				max_word_length=max_word_length)
			# Store data and vocabular[y|ies]
			self.data = tokens
			self.vocabs = vocabs

		elif mode == 'test':
			input_path = os.path.join(data_dir,'test.txt')
			print("Reading %s" % input_path)
			with open(input_path, 'r', encoding='utf-8') as f:
				test_text = f.read()
			
			# Convert text to tokens
			tokens, vocabs = tokenize_func(
				test_data=test_text, 
				save_dir=data_dir, 
				word_markers=self.word_markers,
				max_word_length=max_word_length)
			# Store data
			self.data = tokens
			self.vocabs = vocabs

class BatchLoader(object):
	"""This class is used to build batches of data."""

	def __init__(self, data_loader, batch_size, timesteps, mode='train'):
		"""Standard __init__ function."""
		# Copying pointers
		self.data_loader = data_loader
		self.batch_size = batch_size
		self.timesteps = timesteps
		self.mode = mode

		if self.mode == 'train':
			self.data = self.data_loader.data['train']
		elif self.mode == 'val':
			self.data = self.data_loader.data['val']
		elif self.mode == 'test':
			self.data = self.data_loader.data['test']

		raw_data = []
		for sentence in self.data:
			raw_data += ['<s>'] + sentence + ['</s>']

		self.num_batches = len(raw_data) // (self.batch_size * self.timesteps)
		raw_data = raw_data[:self.num_batches * self.batch_size * self.timesteps]
		self.x_batches = self.data_loader.encode_func(
			np.asarray(raw_data).reshape([self.batch_size, -1, self.timesteps]).transpose([0,2,1]),
			self.data_loader.vocabs,
			self.data_loader.word_markers
		)
		self.y_batches = self.data_loader.encode_func(
			np.asarray(raw_data[1:] + [raw_data[0]]).reshape([self.batch_size, -1, self.timesteps]).transpose([0,2,1]),
			self.data_loader.vocabs,
			self.data_loader.word_markers
		)

		self.reset_pointers()		

	def next_batch(self):
		"""
		Output the next batch.
		Returns:
			x, y, end
		"""
		self.pointer += 1
		if self.pointer > self.num_batches:
			return self.x_batches[:,:,0], self.y_batches[:,:,0], True
		else:
			return self.x_batches[:,:,self.pointer-1], self.y_batches[:,:,self.pointer-1], False
	
	def reset_pointers(self):
		"""
		Function to set up data members for new epoch.
		"""
		self.pointer = 0