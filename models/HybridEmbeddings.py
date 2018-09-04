"""
charCNN-LSTM model that uses a mixture of word- and char-level embeddings.
"""

import tensorflow as tf

class HybridEmbeddings(object):
    def __init__(self, config):
        """ Method for initializing model and constructing graph """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._input = tf.placeholder(tf.float32, [config.batch_size, config.timesteps, config.num_dims], name="input")
            self._output = tf.placeholder(tf.float32, [config.batch_size, config.timesteps], name="correct_output")
            self._states = tf.placeholder(tf.float32, [config.n_layers, config.batch_size, config.num_units], name="initial_states")
            self._lr = tf.placeholder_with_default(1.0, shape=(1,), name="learning_rate")
            
            # TensorFlow variables
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.best_metric = tf.Variable(1000, trainable=False, name="best_metric")

            # Other variables
            self.config = config

            # Create embeddings
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                self.word_embedding = tf.get_variable("word", [config.word_vocab_size, config.word_dims])
                self.char_embedding = tf.get_variable("char", [config.char_vocab_size, config.char_dims])

            # Highway network
            with tf.variable_scope("highway", reuse=tf.AUTO_REUSE):
                x1 = tf.layers.dense(self._input, config.num_dims, activation=tf.nn.relu)
                t1 = tf.layers.dense(self._input, config.num_dims, activation=tf.nn.sigmoid)
                mod_input = x1*t1 + self._input*(1-t1)

            # LSTM network
            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                self.lstm_cells = [tf.contrib.rnn.LSTMCell(
                                            units=config.num_units,
                                            cell_clip=config.lstm_clip,
                                            initializer=tf.contrib.layers.xavier_initializer,
                                            activation=tf.nn.relu,
                                            reuse=tf.AUTO_REUSE
                                            ) for _ in range(config.n_layers)]

                self.initial_states = tf.stack([self.lstm_cells[i].zero_state(config.batch_size, tf.float32) 
                                                for i in range(config.n_layers)],
                                                axis=0)

                inputs = mod_input
                states = tf.unpack(self._states, axis=0)
                self.final_states = []
                for i in range(config.n_layers):
                    with tf.variable_scope("layer_%d" % i, reuse=tf.AUTO_REUSE):
                        inputs, fstate = tf.nn.dynamic_rnn(self.lstm_cells[i], inputs, initial_state=states[i])
                        self.final_states.append(fstate)

                self.final_states = tf.stack(self.final_states, axis=0)
                # reshape 'input', 'target' to [batch_size*timesteps, num_units]
                inputs = tf.reshape(inputs, [-1, config.num_units])
                targets = tf.reshape(self._output, [-1,1]) 

            # Projection and softmax layer
            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                w = tf.get_variable("p_w", [config.num_units, config.word_dims])
                b = tf.get_variable("p_b", config.word_dims)
                output = tf.matmul(inputs, w) + b
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                logits = tf.matmul(output, self.word_embedding, transpose_b=True)
                self.prediction = tf.nn.softmax(logits, axis=1, name="prediction")
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name="loss")

            # Optimizer
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                optim = tf.train.SGDOptimizer(learning_rate=self._lr)
                grads = optim.compute_gradients(self.loss)
                clipped_grads = [(tf.clip_by_global_norm(g, config.grad_clip), v) for g,v in grads]
                self.train_op = optim.apply_gradients(clipped_grads, global_step=self.global_step)
            
            # Evaluation metric
            self.eval_metric = tf.reduce_mean(tf.exp(self.loss))    # trivial PPL

            # Saver
            self.saver = tf.train.Saver(max_to_keep=3)

    def prepare_input(self, x):
        """ Function for preparing input from encoding """
        max_word_length = self.config.max_word_length        

        # Generate one hot matrix of size 
        # [batch_size, timesteps, char_vocab_size, max_word_length]
        lengths = [], idx = []
        for b in x:
            for w in b:
                lengths += [len(w[2])]; idx += w[2]
        
        with self.graph.as_default():
            char_vecs = tf.gather(self.char_embedding, idx)
            char_vecs = tf.split(char_vecs, lengths)
            char_vecs = [tf.pad(cv, [[0, max_word_length-l], [0, 0]], 'CONSTANT', 0)
                            for cv, l in zip(char_vecs, lengths)]
            char_vecs = tf.transpose(tf.stack(char_vecs), [0, 2, 1])
            char_vecs = tf.expand_dims(char_vecs, axis=3)

            # Apply convolution
            with tf.variable_scope("convolution", reuse=tf.AUTO_REUSE):
                feat_vecs = []
                for k, n_ft in zip(self.config.kernel_sizes, self.config.kernel_features):
                    fltr = tf.get_variable("filter_%d" % k, [self.config.char_vocab_size, k, 1, n_ft])
                    out_tensor = tf.nn.conv2d(char_vecs, fltr, strides=[1,1,1,1], padding="SAME")
                    # max-over time pooling
                    pooled_tensor = tf.nn.max_pool(out_tensor, 
                                                    ksize=[1, 1, tf.shape(out_tensor)[2], 1],
                                                    strides=[1, 1, tf.shape(out_tensor)[2], 1],
                                                    padding="VALID"
                                                    )
                    # squeeze the tensor to shape [batch_size*timesteps, n_fts]
                    squeezed_tensor = tf.squeeze(pooled_tensor)
                    feat_vecs.append(squeezed_tensor)
                # concatenate the features obtained
                feature_vec = tf.concat(feat_vecs, axis=1)
                return tf.reshape(feature_vec, [self.config.batch_size, self.config.timesteps, -1])

    def prepare_output(self, y):
        """ Function for preparing correct output tensor from encoding """
        res = np.zeros(y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                res[i,j] = y[i,j][0]
        return res