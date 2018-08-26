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
"""