"""
Functions for encoding tokens.

All functions take list of tokens, and vocabular[y|ies] as input and produce
a list of encodings as output. Each encoding is a dict of the following type:

encoding = {
                0: [list of integers],
                1: [list of integers],
                ...
            }

Here the keys represent embeddings (word-, char-, morpheme- etc.) and the list
of integers represent constituent members from vocabular[y|ies]

OoV words can be handled either here or in the tokenizer.
"""

import numpy as np

""" Default word encoder """
def word_encoder(tokens, vocabs):
    return [{0: vocabs['words'][w], 1:None} for w in tokens]

""" Default character encoder """
def char_encoder(tokens, vocabs):
    return [{0:None, 1: vocabs['chars'][c]} for c in tokens]

""" Custom encoder for LMMRL """
def lmmrl_encoder(tokens, vocabs):
    encoded_arr = np.empty_like(tokens, dtype=object)
    it = np.nditer(tokens, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        w = str(it[0])
        encoding = {1:None}
        if w not in vocabs['words']:
            encoding[0] = vocabs['words']['<unk>']
        else:
            encoding[0] = vocabs['words'][w]
        # encode characters
        encoding[2] = [vocabs['chars']['<w>']]
        for c in w:
            if c not in vocabs['chars']:
                encoding[2].append(vocabs['chars']['<unk>'])
            else:
                encoding[2].append(vocabs['chars'][c])
        encoding[2].append(vocabs['chars']['</w>'])
        encoded_arr[it.multi_index] = encoding
        it.iternext()
    return encoded_arr