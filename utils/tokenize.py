"""
Script containing functions for tokenizing text.

Vocabulary returned by each function is of the following type:
vocab = {
    'words': <word-integer mapping>,
    'chars': <char-integer mapping>,
    ...
}
Here *-integer mapping is a vocabulary of *-space

Data (tokens) returned by each function is split into test, train and val
data = {'test': [] , 'train': [], 'val': []}
"""

import os

""" Generate vocabulary with token as words """
def word_tokenizer(train_data=None, val_data=None, test_data=None, save_dir=None):
    if test_data is None:
        # process training and validation tokens
        tr = train_data.split()
        va = val_data.split()
        # build/load vocabulary
        if save_dir is not None and os.path.exists(os.path.join(save_dir, 'word_vocab.txt')):
            # assume that correct vocab is present
            with open(os.path.join(save_dir, 'word_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            return {'train': tr, 'val': va, 'test': None}, {'words': dict([(w,i) for i,w in enumerate(vocab)]}
        else:
            # construct new vocab
            vocab = {}; cntr = 0
            for w in tr+va:
                if w not in vocab:
                    vocab[w] = cntr
                    cntr = cntr+1
            # save vocabulary
            if save_dir is not None:
                with open(os.path.join(save_dir, 'word_vocab.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join())
    else:
        # process test tokens

""" Generate vocabulary with token as characters """
def char_tokenizer(train_data=None, val_data=None, test_data=None):
    if test_data is None:
        # process training and validation tokens
    else:
        # process test tokens
