"""
Script for training language models.
"""

import tensorflow as tf
import numpy as np
import os, sys, time

def main():
	"""The main method of script."""
	args = parser.parse_args()
	with open(args.config_file, 'r') as stream:
		config = munchify(yaml.load(stream))
	args.save_dir = os.path.join(args.save_dir, args.job_id)
	args.best_dir = os.path.join(args.best_dir, args.job_id)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(args.best_dir):
		os.makedirs(args.best_dir)
	logger.info(args)
	train(
            args.data_dir, 
            token='word', 
            convert=False, 
            batch_size=config.batch_size,
            timesteps = config.timesteps,
            num_epochs = config.num_epochs
            )

# Restores pretrained model from disk 
def restore_model(sess, model, save_dir):
	ckpt = tf.train.get_checkpoint_state(save_dir)
	if ckpt:
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


def train(data_dir, token, convert, batch_size, timesteps, num_epochs, save_dir):
	"""Prepare the data and begin training."""
	# Load the text and vocabulary
	train_data_loader = DataLoader(data_dir, split='train', convert=convert, token=token)
    val_data_loader = DataLoader(data_dir, split='valid', convert=convert, token=token)
	# Prepare batches for training and validation
	train_batch_loader = BatchLoader(train_data_loader, batch_size=batch_size, timesteps=timesteps)
    val_batch_loader = BatchLoader(val_data_loader, batch_size=batch_size, timesteps=timesteps)

    # Run on GPU by default
    cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    cfg_proto.gpu_options.allow_growth = True

	with tf.Session(config=cfg_proto) as sess:
		# Initialize weights
		initializer = tf.random_uniform_initializer(-0.05, 0.05)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			# Build model here
            model = Model()
		steps_done = restore_model(sess, model, save_dir)
		logger.info("Loaded %d completed steps", steps_done)
        # Create summary writer
		train_writer = tf.summary.FileWriter(save_dir + '/logs/', tf.get_default_graph())
        # Finalize graph to prevent memory leaks
		sess.graph.finalize()
        # Find starting epoch
		start_epoch = model.epoch.eval()
        # Start epoch-based training
		for epoch in range(start_epoch, num_epochs):
            # train
			run_epoch(sess, model, args, batch_loader, epoch)
            # validate

def run_epoch(sess, model, batch_loader, epoch, mode='train'):
	"""Run one epoch of training/validation"""
	best_ppl = model.best_ppl.eval()
	last_ppl_update = model.last_ppl_update.eval()
	margin_ppl = model.margin_ppl.eval()
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
        if mode == 'train':
            x, y = batch_loader.next_batch()
            # Feed initial states
            states = sess.run(model.initial_states)
            feed = {model.input_data: x,
                    model.targets: y,
                    model.initial_states: states}
            # carry out the training step
		    train_loss, l1, states, _ = sess.run([model.final_cost,
											 model.cost,
											 model.final_states,
											 model.train_op], feed)
        else if mode == 'val':
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
