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

""" Default word encoder """
def word_encoder(tokens, vocabs):
    return {0: [vocabs['words'][w] for w in tokens], 1:None}

""" Default character encoder """
def char_encoder(tokens, vocabs):
    return {0:None, 1: [vocabs['chars'][c] for c in tokens]}