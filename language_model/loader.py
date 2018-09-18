import os
import torch
from torchtext import data
from torchtext import datasets
import codecs
import re
from nltk import word_tokenize


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


'''
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
'''


class Corpus(object):
    def __init__(self, path, text_field, label_field, batch_size, max_size=40000):
        #self.dictionary = Dictionary()
        self.train = self.read(os.path.join(path, 'train.txt'), text_field, label_field)
        self.valid = self.read(os.path.join(path, 'valid.txt'), text_field, label_field)
        self.test = self.read(os.path.join(path, 'test.txt'), text_field, label_field)
        text_field.build_vocab(self.train, self.valid, self.test, max_size=max_size)
        label_field.build_vocab(self.train, self.valid, self.test, max_size=max_size)
        self.train_iters, self.valid_iters, self.test_iters = data.BucketIterator.splits((self.train, self.valid, self.test), batch_sizes=(batch_size, batch_size, batch_size), device=-1, repeat=False)
        self.dictionary = text_field.vocab.itos
        # Old codes
        #self.train = self.tokenize(os.path.join(path, 'train.txt'))
        #self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def read(self, path, text_field, label_field):
        fields = [('text', text_field), ('label', label_field)]
        texts = self.read_dataset(path)
        examples = []
        for text in texts:
            if len(text) < 3:
                continue
            #_data = [text[:-1], text[1:]]
            _data = [text, text]
            _fields = [('text', text_field), ('label', label_field)]
            a = data.Example()
            #print _data
            examples.append(a.fromlist(_data, _fields))
        ret_data = data.Dataset(examples, fields)
        ret_data.sort_key = lambda x: -1 * len(x.text)
        return ret_data

    def read_dataset(self, path):
        texts = []
        for line in codecs.open(path, 'r', 'utf8'):
            line = zero_digits(line.rstrip())
            line = word_tokenize(line)
            #line = line.split(' ')
            text = [w.lower() for w in line]
            texts.append(text)
        return texts

    '''
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    '''
