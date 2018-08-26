"""
charCNN-LSTM model that uses a mixture of word- and char-level embeddings.
"""

class HybridEmbeddings(object):
    def __init__(self, config):
        """ Method for initializing model and constructing graph """