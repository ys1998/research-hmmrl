""" Scripts for analysing training data """

import matplotlib.pyplot as plt
import sys, os
import numpy as np

def main():
    tr_path = sys.argv[1]
    if not os.path.exists(tr_path):
        print('File not found.')
        exit()
    freq = {}
    with open(tr_path, 'r', encoding='utf-8') as f:
        sentences = f.read().split('\n')

    sentence_lens = {}
    for s in sentences:
        if len(s) in sentence_lens:
            sentence_lens[len(s)] += 1
        else:
            sentence_lens[len(s)] = 1

    plt.figure()
    plt.plot(sentence_lens.keys(), np.cumsum(list(sentence_lens.values()))/sum(list(sentence_lens.values())))
    plt.title('Sentence lengths')
    plt.show(block=False)

    for s in sentences:
        words = s.split(' ')
        for w in words:
            # trigrams
            for i in range(len(w)-3):
                if w[i:i+3] not in freq:
                    freq[w[i:i+3]] = 1
                else:
                    freq[w[i:i+3]] = freq[w[i:i+3]] + 1
            # bigrams
            for i in range(len(w)-2):
                if w[i:i+2] not in freq:
                    freq[w[i:i+2]] = 1
                else:
                    freq[w[i:i+2]] = freq[w[i:i+2]] + 1
    bigrams = []; trigrams = []
    for k,v in freq.items():
        if(len(k)==2):
            bigrams.append((k,v))
        else:
            trigrams.append((k,v))

    bigrams = sorted(bigrams, key=lambda x: x[1], reverse=True)
    trigrams = sorted(trigrams, key=lambda x: x[1], reverse=True)

    # bigrams
    plt.figure();
    x, y = zip(*bigrams)
    plt.plot(list(range(len(y))),y, 'bo')
    plt.xticks(np.arange(0, len(y), round(len(y)/100)), rotation='vertical')
    plt.yticks(np.arange(0, max(y), round(max(y)/50)))
    plt.title("Bigrams (%d) %s"%(len(bigrams), sys.argv[1]))
    plt.xlabel('Bigram index')
    plt.ylabel('Frequency')
    plt.show(block=False)

    # trigrams
    plt.figure()
    x, y = zip(*trigrams)
    plt.plot(list(range(len(y))),y, 'bo')
    plt.xticks(np.arange(0, len(y), round(len(y)/100)), rotation='vertical')
    plt.yticks(np.arange(0, max(y), round(max(y)/50)))
    plt.title("Trigrams (%d) %s"%(len(trigrams), sys.argv[1]))
    plt.xlabel('Trigram index')
    plt.ylabel('Frequency')
    plt.show(block=False)

    # sentence stats
    pure = 0; impure = 0; only_num = 0; english_chars_too = 0
    for s in sentences:
        eng_nums = 0; eng_chars = 0; non_hindi = 0
        for w in s.split():
            for c in w:
                if (c >= '\u0900' and c <= '\u097F') or (c >= '\u0020' and c <= '\u007E'):
                    if c >= '\u0030' and c <= '\u0039':
                        eng_nums+=1
                    elif (c >= '\u0041' and c <= '\u005A') or (c >= '\u0061' and c <= '\u007A'):
                        eng_chars+=1
                    else:
                        pass #punctuation
                else:
                    non_hindi += 1
                    break
            if non_hindi > 0:
                break
        if non_hindi > 0:
            impure+=1
        elif eng_nums > 0 and eng_chars == 0:
            only_num+=1
        elif eng_chars > 0:
            english_chars_too+=1
        else:
            pure+=1

    plt.figure()
    plt.pie([pure, impure, only_num, english_chars_too], 
            labels=['Pure Hindi', 'With non-Hindi chars', 'Only English numbers', 'With English chars'],
            autopct=lambda x: "%.2f (%d)"%(x, round(x*0.01*len(sentences)))
            )
    plt.title('Type of sentences %s'%sys.argv[1])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python analysis.py [train.txt]')
        exit()
    main()