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

""" Default function for generating vocabulary with token as words """
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
            return {'train': tr, 'val': va, 'test': None}, {'words': {w:i for i,w in enumerate(vocab)}}
        else:
            # construct new vocab
            vocab = {}; cntr = 0
            for w in tr+va:
                if w not in vocab:
                    vocab[w] = cntr
                    cntr = cntr+1
            # add <unk> token
            vocab['<unk>'] = cntr
            # save vocabulary
            if save_dir is not None:
                with open(os.path.join(save_dir, 'word_vocab.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(vocab.keys(), key=lambda x: vocab[x])))
            return  {'train': tr, 'val': va, 'test': None}, {'words': vocab}
    else:
        # process test tokens
        if save_dir is None or not os.path.exists(os.path.join(save_dir, 'word_vocab.txt')):
            print("Could not find vocabulary file.")
        else:
            with open(os.path.join(save_dir, 'word_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            # generate mapping
            vocab = {w:i for i,w in enumerate(vocab)}
            # process OoV words
            td = []
            for w in test_data:
                if w in vocab:
                    td += [w]
                else:
                    td += ['<unk>']
            return  {'train':None, 'val':None, 'test':td}, {'words':vocab}
            
""" Default function for generating vocabulary with token as characters """
def char_tokenizer(train_data=None, val_data=None, test_data=None, save_dir=None):
    if test_data is None:
        # process training and validation tokens
        tr = list(train_data.replace('\n', ' '))
        va = list(val_data.replace('\n', ' '))
        # build/load vocabulary
        if save_dir is not None and os.path.exists(os.path.join(save_dir, 'char_vocab.txt')):
            # assume that correct vocab is present
            with open(os.path.join(save_dir, 'char_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            return  {'train': tr, 'val': va, 'test': None}, {'chars': {c:i for i,c in enumerate(vocab)}}
        else:
            # construct new vocab
            vocab = {}; cntr = 0
            for c in tr+va:
                if c not in vocab:
                    vocab[c] = cntr
                    cntr = cntr+1
            # add <unk> token
            vocab['<unk>'] = cntr
            # save vocabulary
            if save_dir is not None:
                with open(os.path.join(save_dir, 'char_vocab.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(vocab.keys(), key=lambda x: vocab[x])))
            return  {'train': tr, 'val': va, 'test': None}, {'chars': vocab}
    else:
        # process test tokens
        if save_dir is None or not os.path.exists(os.path.join(save_dir, 'char_vocab.txt')):
            print("Could not find vocabulary file.")
        else:
            with open(os.path.join(save_dir, 'char_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            # generate mapping
            vocab = {c:i for i,c in enumerate(vocab)}
            # process OoV words
            td = []
            for c in test_data:
                if c in vocab:
                    td += [c]
                else:
                    td += ['<unk>']
            return  {'train':None, 'val':None, 'test':td}, {'words':vocab}

""" Custom tokenizer for LMMRL """
def lmmrl_tokenizer(train_data=None, val_data=None, test_data=None, save_dir=None):
    if test_data is None:
        # process training and validation tokens
        tr = [x.split() for x in train_data.split('\n')]
        va = [x.split() for x in val_data.split('\n')]
        # build/load vocabulary
        vocabs = {}
        if save_dir is not None and os.path.exists(os.path.join(save_dir, 'word_vocab.txt')):
            # assume that correct vocab is present
            with open(os.path.join(save_dir, 'word_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split() 
            vocabs['words'] = {w:i for i,w in enumerate(vocab)}
        else:
            # construct new vocab and freq
            vocab = {}; cntr = 0; freq = {}
            for s in tr:    # consider tokens present only in training set
                for w in s:
                    if w not in vocab:
                        vocab[w] = cntr
                        freq[w] = 1
                        cntr = cntr+1
                    else:
                        freq[w] = freq[w] + 1
            vocab['<unk>'] = cntr # add <unk> token
            vocab['<pad>'] = cntr+1 # add pad token
            vocab['<s>'] = cntr+2
            vocab['</s>'] = cntr+3 # add sentence termination tokens
            freq['<unk>'] = freq['<pad>'] = freq['<s>'] = freq['</s>'] = 1
            vocabs['words'] = vocab
            # save vocabulary
            if save_dir is not None:
                with open(os.path.join(save_dir, 'word_vocab.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(vocab.keys(), key=lambda x: vocab[x])))
                with open(os.path.join(save_dir, 'word_freq.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join([str(freq[x]) for x in sorted(vocab.keys(), key=lambda x: vocab[x])]))
        # build char vocab
        tr_c = list(train_data.replace('\n', ' '))
        if save_dir is not None and os.path.exists(os.path.join(save_dir, 'char_vocab.txt')):
            # assume that correct vocab is present
            with open(os.path.join(save_dir, 'char_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split() 
            vocabs['chars'] = {w:i for i,w in enumerate(vocab)}
        else:
            # construct new vocab
            vocab = {}; cntr = 0
            for c in tr_c:  # consider tokens present only in training set
                if c not in vocab:
                    vocab[c] = cntr
                    cntr = cntr+1
            # add <unk> token
            vocab['<unk>'] = cntr
            vocabs['chars'] = vocab
            # save vocabulary
            if save_dir is not None:
                with open(os.path.join(save_dir, 'char_vocab.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(vocab.keys(), key=lambda x: vocab[x])))

        # data is in the form of a list of sentences
        return  {'train':tr, 'val':va, 'test':None}, vocabs
                
    else:
        # process test tokens
        if save_dir is None or not os.path.exists(os.path.join(save_dir, 'word_vocab.txt')) or not os.path.exists(os.path.join(save_dir, 'char_vocab.txt')):
            print("Could not find vocabulary file.")
        else:
            vocabs = {}
            with open(os.path.join(save_dir, 'word_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            # generate mapping
            vocabs['words'] = {w:i for i,w in enumerate(vocab)}
            with open(os.path.join(save_dir, 'char_vocab.txt'), 'r', encoding='utf-8') as f:
                vocab = f.read().split()
            # generate mapping
            vocabs['chars'] = {w:i for i,w in enumerate(vocab)}

            return  {'train':None, 'val':None, 'test':[x.split() for x in test_data.split('\n')]}, vocabs
