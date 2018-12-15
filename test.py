"""
Script for testing language models.
"""

import tensorflow as tf
import numpy as np
import os, yaml
from munch import munchify

from utils.arguments import test_parser as parser
from utils.loader import DataLoader, BatchLoader
from utils.tokenize import lmmrl_tokenizer
from utils.encode import lmmrl_encoder

# Import model
# from models.HybridEmbeddings import HybridEmbeddings as Model
from models.PoolingWindow import Model

tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)

def main():
	"""The main method of script."""
	args = parser.parse_args()
	mode = None

	if not os.path.exists(args.model_dir):
		print("Model not found.")
		exit()
	if not os.path.exists(args.config_file):
		print("Configuration file not found.")
		exit()
	if args.test_dir is None:
		if not os.path.exists(args.prior_file):
			print("Prior file doesn't exist.")
			exit()
		else:
			mode = 1
	else:
		if not os.path.exists(args.test_dir):
			print("Test directory does not exist.")
			exit()
		else:
			mode = 2

	with open(args.config_file, 'r') as stream:
		config = munchify(yaml.load(stream))
	if mode == 1:
		generate()
	else:
		test(config, args.model_dir, args.test_dir)

# Restores pretrained model from disk 
def restore_model(sess, model, save_dir):
	ckpt = tf.train.get_checkpoint_state(save_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
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

def generate():
	pass

def test(config, model_dir, test_dir):
	data_loader = DataLoader(test_dir, mode='test', tokenize_func=lmmrl_tokenizer, encode_func=lmmrl_encoder)
	batch_loader = BatchLoader(data_loader, batch_size=config.batch_size, timesteps=config.timesteps, mode='test')
	
	cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
	cfg_proto.gpu_options.allow_growth = True

	# Load word frequency information
	with open(os.path.join(test_dir, 'word_freq.txt'), encoding='utf-8') as f:
		freq = f.read().split()
		config['freq'] = freq
	config.save_dir = model_dir

	model = Model(config)

	with tf.Session(config=cfg_proto, graph=model.graph) as sess:
		# Restore model/Initialize weights
		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			_ = restore_model(sess, model, model_dir)
		
		print("Model restored from %s" % model_dir)
		# Finalize graph to prevent memory leakage
		sess.graph.finalize()

		# Prepare loader
		batch_loader.reset_pointers()
		# Start from an empty RNN state
		init_states = sess.run(model.initial_states)
		states = init_states

		acc_loss = np.zeros(batch_loader.batch_size)
		acc_lengths = np.zeros(batch_loader.batch_size)
		sentence_ppls = []

		end_epoch = False
		b = 1
		while not end_epoch:
			x, y, lengths, reset, end_epoch = batch_loader.next_batch()
			if end_epoch:
				break
			loss, _ = model.forward(sess, x, y, states, lengths, mode='test')
			# accumulate evaluation metric here
			acc_loss += loss*lengths
			acc_lengths += lengths
			print("Batch = %d, Average loss = %.4f" % (b, np.mean(loss)))
			b += 1
			for i in range(len(reset)):
				if reset[i] == 1.0:
					states[:,:,i,:] = init_states[:,:,i,:]
					sentence_ppls.append(np.exp(acc_loss[i]/acc_lengths[i]))
					acc_loss[i] = acc_lengths[i] = 0.0

		# find metric from accumulated metrics of sentences
		final_metric = np.mean(sentence_ppls)
		print("Sentence-wise perplexities\n", sentence_ppls)
		print("(Averaged) Evaluation metric = %.4f" % final_metric)

if __name__ == '__main__':
	main()