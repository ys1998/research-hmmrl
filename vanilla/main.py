#!/usr/bin/python
"""The primary script to execute the tensorflow models."""
from __future__ import print_function
from munch import munchify
from six.moves import cPickle
import json
import logging
import os
import sys
import time
import yaml
import numpy as np
import tensorflow as tf

from arguments import parser
from model import Model
from processor import BatchLoader, DataLoader, eval_loader
import adaptive

logging.basicConfig(
	stream=sys.stdout,
	format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
	level=logging.INFO
)
logger = logging.getLogger(__name__)

tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)


def main():
	"""The main method of script."""
	args = parser.parse_args()
	with open(args.config_file, 'r') as stream:
		args.config = munchify(yaml.load(stream))
	args.save_dir = os.path.join(args.save_dir, args.job_id)
	args.best_dir = os.path.join(args.best_dir, args.job_id)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(args.best_dir):
		os.makedirs(args.best_dir)
	logger.info(args)
	if args.mode == 'train':
		train(args)
	elif args.mode == 'test' or args.mode == 'valid':
		test(args)
	elif args.mode == 'generate':
		generate(args)


def generate(args):
	args.config.timesteps = 1
	data_loader = DataLoader(args)
	if args.device == "gpu":
		cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
		cfg_proto.gpu_options.allow_growth = True
	else:
		cfg_proto = None
	with tf.Session(config=cfg_proto) as sess:

		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = Model(args, args.config.batch_size, mode='train')
		steps_done = initialize_weights(sess, model, args, mode='test')
		with tf.variable_scope("model", reuse=True, initializer=initializer):
			model = Model(args, batch_size=1, mode='eval')

		logger.info("loaded %d completed steps", steps_done)
		config = json.loads(args.gen_config)
		states = sess.run(model.initial_states)
		# First feed in the prior letters
		probs = None
		for i in config['prior'].split():
			feed = {
				model.input_data: np.array([[data_loader.vocab[i]]]),
				model.initial_states: states
			}
			[states, probs] = sess.run([model.final_states, model.probs], feed)
		# Now, time to begin the sampling process

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t, np.random.rand(1) * s)))

		# Weird construct to optimize code
		prior_length = len(config['prior'].split())
		output = [' '] * (config['length'] + prior_length)
		for i in range(prior_length):
			output[i] = config['prior'].split()[i]

		for i in range(config['length']):
			if i % 100 == 0:
				print("%d out of %d generated" % (i, config['length']))
			token = weighted_pick(np.squeeze(probs))
			if token == len(np.squeeze(probs)):
				token = token - 1
			output[i + prior_length] = data_loader.rev_vocab[token]
			feed = {
				model.input_data: np.array([[token]]),
				model.initial_states: states
			}
			[states, probs] = sess.run([model.final_states, model.probs], feed)

		output = ' '.join(output)
		output = output.replace('</s>', '\n')
		output = output + "\n"

		with open(os.path.join(args.save_dir, "generate_{0}.txt".format(args.job_id)), 'w') as f:
			f.write(output)


def initialize_weights(sess, model, args, mode):
	ckpt = tf.train.get_checkpoint_state(args.save_dir)
	ckpt_best = tf.train.get_checkpoint_state(args.best_dir)
	if mode == 'test' and ckpt_best:
		logger.info("Reading best model parameters from %s", ckpt_best.model_checkpoint_path)
		model.saver.restore(sess, ckpt_best.model_checkpoint_path)
		steps_done = int(ckpt_best.model_checkpoint_path.split('-')[-1])
		# Since local variables are not saved
		sess.run([
			tf.local_variables_initializer()
		])
	elif mode == 'train' and ckpt:
		logger.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
		model.saver.restore(sess, ckpt.model_checkpoint_path)
		steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
		# Since local variables are not saved
		sess.run([
			tf.local_variables_initializer()
		])
	else:
		steps_done = 0
		sess.run([
			tf.global_variables_initializer(),
			tf.local_variables_initializer()
		])
	return steps_done


def evaluate(sess, model, eval_data, args, calculate_prob=False, rev_vocab=None):
	"""Calculate perplexity after every epoch."""
	states = sess.run(model.initial_states)
	total_loss = 0.0
	prob_output = ""
	eval_x, eval_y, eval_len = eval_data['x'], eval_data['y'], eval_data['len']
	for i in range(eval_x.shape[0]):
		# Need to pass L1 to get evaluation perplexity
		feed = {
			model.input_data: eval_x[i:i + 1, :],
			model.targets: eval_y[i:i + 1, :],
			model.initial_states: states
		}
		if calculate_prob is True:
			[states, loss, probs] = sess.run([model.final_states, model.loss, model.probs], feed)
			total_loss += loss.sum()
			for j in range(len(probs)):
				position = i * args.config.timesteps + j
				if position >= eval_len - 1:
					continue
				token = eval_y[i][j]
				prob_output += rev_vocab[token] + " " + str(probs[j, token]) + "\n"
		else:
			[states, loss] = sess.run([model.final_states, model.loss], feed)
			total_loss += loss.sum()

	# need to subtract off loss from padding tokens
	extra_tokens = (args.config.timesteps - eval_len % args.config.timesteps) % args.config.timesteps + 1
	total_loss -= loss[-extra_tokens:].sum()
	avg_entropy = total_loss / eval_len
	ppl = np.exp(avg_entropy)

	if calculate_prob is True:
		return ppl, prob_output
	else:
		return ppl


def test(args):
	data_loader = DataLoader(args)
	if args.device == "gpu":
		cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
		cfg_proto.gpu_options.allow_growth = True
	else:
		cfg_proto = None
	with tf.Session(config=cfg_proto) as sess:

		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = Model(args, args.config.batch_size, mode='train')
		steps_done = initialize_weights(sess, model, args, mode='test')

		with tf.variable_scope("model", reuse=True, initializer=initializer):
			model_eval = Model(args, batch_size=1, mode='eval')

		logger.info("loaded %d completed steps", steps_done)
		test_data = {}
		test_data['x'], test_data['y'], test_data['len'] = eval_loader(args, data_loader.vocab, split=args.mode)
		ppl, prob_output = evaluate(
			sess, model_eval, test_data, args, calculate_prob=True, rev_vocab=data_loader.rev_vocab
		)
		with open(os.path.join(args.save_dir, "probs_{0}_{1}.txt".format(args.mode,args.job_id)), 'w') as f:
			f.write(prob_output)
		logger.info("Perplexity is %.4f", ppl)


def train(args):
	"""Prepare the data and begins training."""
	# Load the text and vocabulary
	data_loader = DataLoader(args)
	# Prepare batches for training
	batch_loader = BatchLoader(args, data_loader)

	if args.device == "gpu":
		cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
		cfg_proto.gpu_options.allow_growth = True
	else:
		cfg_proto = None
	with tf.Session(config=cfg_proto) as sess:
		# Build training model, load old weights
		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = Model(args, args.config.batch_size, mode='train')
		steps_done = initialize_weights(sess, model, args, mode='train')
		logger.info("loaded %d completed steps", steps_done)

		# Reusing weights for evaluation model
		with tf.variable_scope("model", reuse=True, initializer=initializer):
			model_eval = Model(args, batch_size=1, mode='eval')
		valid_data = {}
		valid_data['x'], valid_data['y'], valid_data['len'] = eval_loader(args, data_loader.vocab, split='valid')
		batch_loader.eval_data = valid_data
		train_writer = tf.summary.FileWriter(args.save_dir + '/logs/', tf.get_default_graph())
		# Making the graph read-only to prevent memory leaks
		# https://stackoverflow.com/documentation/tensorflow/3883/how-to-debug-a-memory-leak-in-tensorflow/13426/use-graph-finalize-to-catch-nodes-being-added-to-the-graph#t=201612280201558374055
		sess.graph.finalize()
		start_epoch = model.epoch.eval()
		for epoch in range(start_epoch, args.config.num_epochs):
			run_epoch(sess, model, model_eval, args, batch_loader, epoch)

def run_epoch(sess, model, model_eval, args, batch_loader, epoch):
	"""Run one epoch of training."""
	best_ppl = model.best_ppl.eval()
	last_ppl_update = model.last_ppl_update.eval()
	margin_ppl = model.margin_ppl.eval()
	adaptive_loss = getattr(adaptive, args.loss_mode)
	# Reset batch pointer back to zero
	batch_loader.reset_batch_pointer()
	# Start from an empty RNN state
	states = sess.run(model.initial_states)

	start_batch = model.global_step.eval() % batch_loader.num_batches
	if start_batch != 0:
		logger.info("Starting from batch %d / %d", start_batch, batch_loader.num_batches)
		batch_loader.pointer += start_batch

	for b in range(start_batch, batch_loader.num_batches):
		start = time.time()
		l1 = adaptive_loss(epoch, b, args=args)
		sess.run(model.l1_assign, feed_dict={model.l1_new: l1})
		x, y = batch_loader.next_batch(l1)
		# With probability 0.01 feed the initial state
		if np.random.randint(1, 101) <= 1:
			states = sess.run(model.initial_states)

		feed = {model.input_data: x,
				model.targets: y,
				model.initial_states: states}
		train_loss, l1, states, _ = sess.run([model.final_cost,
											 model.cost,
											 model.final_states,
											 model.train_op], feed)
		end = time.time()
		# print the result so far on terminal
		batch_num = epoch * batch_loader.num_batches + b
		total_num = args.config.num_epochs * batch_loader.num_batches
		logger.info("Epoch %d, %d / %d. Loss - %.4f, Time - %.2f", epoch, batch_num, total_num, train_loss, end - start)

		# Save after `args.eval_freq` batches or at the very end
		if batch_num != 0 and (batch_num % args.config.eval_freq == 0 or b == batch_loader.num_batches - 1):
			ppl = evaluate(sess, model_eval, batch_loader.eval_data, args)
			logger.info("Perplexity after %d steps - %.4f", batch_num, ppl)

			# Update rules for best_ppl / training schedule
			logger.info("Best ppl is %.4f, ppl < best_ppl is %s", model.best_ppl.eval(), str(ppl < best_ppl))
			if ppl < best_ppl:
				logger.info("Saving Best Model")
				# Storing perplexity and in TensorFlow variable and Python variable
				best_ppl = ppl
				sess.run(model.best_ppl_assign, feed_dict={model.best_ppl_new: ppl})
				if margin_ppl - ppl > args.config.margin_ppl:
					# In the case there has been a perplexity change of more than `margin_ppl`
					# update the `last_ppl_update` and `margin_ppl` values
					# This indicates a "significant" change in ppl
					logger.info("Updating margin_ppl, Change of %.4f", margin_ppl - ppl)
					last_ppl_update = batch_num
					margin_ppl = ppl
					sess.run(model.last_ppl_update_assign, feed_dict={model.last_ppl_update_new: batch_num})
					sess.run(model.margin_ppl_assign, feed_dict={model.margin_ppl_new: ppl})
				# Saving the best model
				checkpoint_path = os.path.join(args.best_dir, "lm.ckpt")
				model.best_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
			# elif batch_num - last_ppl_update > args.config.eval_freq * 30:
			#     logger.info("Decaying Learning Rate")
			#     sess.run(model.lr_decay)
			#     # Updating `last_ppl_update` value to prevent continuous decay, keeping same `margin_ppl`
			#     last_ppl_update = batch_num
			#     sess.run(model.last_ppl_update_assign, feed_dict={model.last_ppl_update_new: batch_num})
			# Learning rate decay schedule
			else:
				# Decay learning rate whenever ppl is greater than best_ppl so far
				sess.run(model.lr_decay)
				logger.info("decaying lr after %d epochs to %.4f" % (model.epoch.eval(), model.lr.eval()))

			checkpoint_path = os.path.join(args.save_dir, "lm.ckpt")
			model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

	sess.run(model.epoch_incr)

if __name__ == '__main__':

	main()
