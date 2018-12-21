"""
Script for training language models.
"""

import tensorflow as tf
import numpy as np
import os, sys, time, yaml, logging
from munch import munchify

# Import model
from models.HybridEmbeddings import HybridEmbeddings as Model
# from models.PoolingWindow import Model
# Import other utilities
from utils.arguments import train_parser as parser
from utils.loader import DataLoader, BatchLoader
# Import tokenizer and encoder
from utils.tokenize import lmmrl_tokenizer
from utils.encode import lmmrl_encoder

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
		config = munchify(yaml.load(stream))
		# print(config)
	args.save_dir = os.path.join(args.save_dir, args.job_id)
	args.best_dir = os.path.join(args.best_dir, args.job_id)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(args.best_dir):
		os.makedirs(args.best_dir)
	train(args.data_dir, args.save_dir,	args.best_dir, config)

# Restores pretrained model from disk 
def restore_model(sess, model, save_dir):
	ckpt = tf.train.get_checkpoint_state(save_dir)
	if ckpt:
		logger.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
		model.saver.restore(sess, ckpt.model_checkpoint_path)
		steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
		# Since local variables are not saved
		sess.run([model.local_initializer])
	else:
		steps_done = 0
		sess.run([
			model.global_initializer,
			model.local_initializer
		])
	return steps_done 


def train(data_dir, save_dir, best_dir, config):
	"""Prepare the data and begin training."""
	# Create variables
	batch_size = config.batch_size
	timesteps = config.timesteps
	num_epochs = config.epochs
	# Load the text and vocabulary
	data_loader = DataLoader(
		data_dir, 
		mode='train', 
		tokenize_func=lmmrl_tokenizer, 
		encode_func=lmmrl_encoder, 
		word_markers=config.include_word_markers,
		max_word_length=config.max_word_length
	)
	# Prepare batches for training and validation
	train_batch_loader = BatchLoader(data_loader, batch_size=batch_size, timesteps=timesteps, mode='train')
	val_batch_loader = BatchLoader(data_loader, batch_size=batch_size, timesteps=timesteps, mode='val')

	# update vocabulary sizes
	config.word_vocab_size = len(data_loader.vocabs['words'])
	config.char_vocab_size = len(data_loader.vocabs['chars'])

	# Run on GPU by default
	cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
	cfg_proto.gpu_options.allow_growth = True

	##########################################################################
	# Load word frequency information
	##########################################################################
	with open(os.path.join(data_dir, 'word_freq.txt'), encoding='utf-8') as f:
		freq = f.read().split()
		config['freq'] = freq
	##########################################################################

	# Create model
	config.save_dir = save_dir
	model = Model(config)

	with tf.Session(config=cfg_proto, graph=model.graph) as sess:
		# Restore model/Initialize weights
		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			steps_done = restore_model(sess, model, save_dir)
		
		logger.info("Loaded %d completed steps", steps_done)

		# Find starting epoch
		start_epoch = model.epoch_cntr.eval()

		# Start epoch-based training
		lr = config.initial_learning_rate
		
		# Finalize graph to prevent memory leakage
		sess.graph.finalize()
		last_val_ppl = 10000
		
		for epoch in range(start_epoch, num_epochs):
			logger.info("Epoch %d / %d", epoch+1, num_epochs)
			# train
			run_epoch(sess, model, train_batch_loader, 'train', save_dir=save_dir, lr=lr)
			# fine-tune after every epoch
			model.fine_tune(sess)
			# validate
			val_ppl = run_epoch(sess, model, val_batch_loader, 'val', best_dir=best_dir)
			# update learning rate conditionally
			if val_ppl >= last_val_ppl:
				lr *= config.lr_decay
				logger.info("Decaying learning rate to %.4f", lr)
			last_val_ppl = val_ppl
			# increment epoch
			sess.run([model.incr_epoch])

def run_epoch(sess, model, batch_loader, mode='train', save_dir=None, best_dir=None, lr=None):
	"""Run one epoch of training."""
	# Prepare loader for new epoch
	batch_loader.reset_pointers()
	# Start from an empty RNN state
	init_states = sess.run(model.initial_states)
	states = init_states

	if mode == 'val':
		acc_loss = 0.0

	end_epoch = False
	b = 1
	while not end_epoch:
		x, y, end_epoch = batch_loader.next_batch()
		if end_epoch:
			break
		if mode == 'train':
			start = time.time()
			# can update the learning rate here, if required
			# lr = 1.0

			loss, states = model.forward(sess, x, y, states, lr, mode)
			end = time.time()
			# print the result so far on terminal
			logger.info("Batch %d, Loss - %.4f, Time - %.2f", b, loss, end - start)

		elif mode == 'val':
			loss = model.forward(sess, x, y, states, mode=mode)
			acc_loss += loss
		
		b += 1

		
	# After epoch is complete
	batch_loader.reset_pointers()
	if mode == 'val':
		# find metric from accumulated metrics of sentences
		final_metric = np.exp(acc_loss/b)
		best_metric = model.best_metric.eval()
		logger.info("(Averaged) Evaluation metric = %.3f", final_metric)
		logger.info("Best metric = %.3f", best_metric)
		if final_metric < best_metric:
			logger.info("Metric improved, saving best model")
			# Store best metric in the model
			sess.run([model.update_best_metric], feed_dict={model.new_best_metric: final_metric})
			# Save the model
			checkpoint_path = os.path.join(best_dir, "lm.ckpt")
			model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
		return final_metric
	elif mode == 'train':
		# Save the model
		checkpoint_path = os.path.join(save_dir, "lm.ckpt")
		model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
		
if __name__ == '__main__':
	main()