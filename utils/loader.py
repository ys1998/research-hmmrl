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
			tokens, vocabs = tokenize_func(test_data=test_text, save_dir=data_dir)
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

		self.reset_pointers()		

	def next_batch(self):
		"""
		Output the next batch.
		Returns:
			x, y, lengths, reset, end
		"""
		
		if self.update_sentences:
			idx = [min(x, self.num_batches-1) for x in self.batch_pointer]
			self.sentences = self.data[self.indices[idx, np.arange(self.batch_size, dtype=int)]]
			for i in range(len(self.sentences)):
				self.sentences[i] = ['<s>'] + self.sentences[i] + ['</s>']
			self.update_sentences = False
		
		reset = self.reset_reqd
		x = y = np.empty([self.batch_size, self.timesteps], dtype=object)
		lengths = np.zeros(self.batch_size, dtype=int)
		batch_over = [False]*self.batch_size

		for i in range(self.batch_size):
			if self.batch_pointer[i] == self.num_batches:
				batch_over[i] = True
				x[i] = y[i] = ['<pad>']*self.timesteps
				self.reset_reqd[i] = 1.0
				continue

			l = len(self.sentences[i][self.word_pointer[i]:-1])
			if l <= self.timesteps:
				x[i] = self.sentences[i][self.word_pointer[i]:-1] + ['<pad>']*(self.timesteps - l)
				y[i] = self.sentences[i][self.word_pointer[i]+1:] + ['<pad>']*(self.timesteps - l)
				lengths[i] = l
				self.reset_reqd[i] = 1.0
				self.batch_pointer[i] += 1
				self.update_sentences = True
				self.word_pointer[i] = 0
			else:
				x[i] = self.sentences[i][self.word_pointer[i]:self.timesteps]
				y[i] = self.sentences[i][self.word_pointer[i]+1:self.timesteps+1]
				lengths[i] = self.timesteps
				self.reset_reqd[i] = 0.0
				self.word_pointer[i] += self.timesteps

		x = self.data_loader.encode_func(x, self.data_loader.vocabs)
		y = self.data_loader.encode_func(y, self.data_loader.vocabs)

		return x, y, lengths, reset, all(batch_over)
	
	def reset_pointers(self):
		"""
		Function to set up data members for new epoch.
		"""
		if self.mode == 'train':
			self.data = self.data_loader.data['train']
		elif self.mode == 'val':
			self.data = self.data_loader.data['val']
		elif self.mode == 'test':
			self.data = self.data_loader.data['test']

		self.data = np.asarray(self.data)
		# shuffle training data
		np.random.shuffle(self.data)
		# calculate number of groups
		self.num_batches = len(self.data) // self.batch_size
		# neglect trailing sentences
		self.data = self.data[:self.num_batches*self.batch_size]
		# create groups of indices
		self.indices = np.random.choice(len(self.data), [self.num_batches, self.batch_size], replace=False)
		# create independent pointers for each member of a batch
		self.batch_pointer = np.zeros(self.batch_size, dtype=int)
		# create pointer for word of each batch-member
		self.word_pointer = np.zeros(self.batch_size, dtype=int)
		# boolean array indicating whether states need to be reset
		self.reset_reqd = np.zeros(self.batch_size)
		# boolean to indicate whether to update sentences
		self.update_sentences = True


