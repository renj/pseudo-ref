import sys
sys.path.append('../pytorch-examples/word_language_model/')
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.patches as patches

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torchtext import data
import loader

def load_corpus(path):
    batch_size = 20
    text_field = data.Field(lower=True)
    label_field = data.Field(lower=True)
    return loader.Corpus(path, text_field, label_field, batch_size)

def get_hiddens(model, dictionary, sent):
    feature = torch.from_numpy(np.array([dictionary[w] for w in sent]))
    feature = Variable(feature.view(-1, 1).cuda(model.device))
    emb = model.encoder(feature)
    output, hidden = model.rnn(emb)
    return output

def get_matrix(model, dictionary, s0, s1):
    if type(s0) is str:
        s0 = s0.split(' ')
    if type(s0) is str:
        s1 = s1.split(' ')
    h0 = get_hiddens(model, dictionary, s0)
    h1 = get_hiddens(model, dictionary, s1)
    hsize = model.rnn.hidden_size
    if model.rnn.bidirectional:
        hsize *= 2
    h0 = [h[0].cpu().data.numpy().reshape(hsize) for h in h0]
    h1 = [h[0].cpu().data.numpy().reshape(hsize) for h in h1]
    m = []
    for _h0 in h0:
        _m = []
        for _h1 in h1:
            #_m += [sum(_h0 * _h1)]
            #_m += [euclid_dist(_h0, _h1)]
            _m += [cosine(_h0, _h1)]
        m += [_m]
    m = 1 - np.array(m)
    return m

def get_matrix_from_hidden(h0, h1):
    #hsize = model.rnn.hidden_size
    #if model.rnn.bidirectional:
    #    hsize *= 2
    #h0 = [h[0].cpu().data.numpy().reshape(hsize) for h in h0]
    #h1 = [h[0].cpu().data.numpy().reshape(hsize) for h in h1]
    m = []
    for _h0 in h0:
        _m = []
        for _h1 in h1:
            #_m += [sum(_h0 * _h1)]
            #_m += [euclid_dist(_h0, _h1)]
            _m += [cosine(_h0, _h1)]
        m += [_m]
    m = 1 - np.array(m)
    return m

def draw(m, s0, s1, alignment=[]):
    #fig, ax = plt.subplots()
    if type(s0) is not list:
        s0 = s0.split(' ')
    if type(s1) is not list:
        s1 = s1.split(' ')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.imshow(m, cmap=plt.cm.Blues, interpolation='nearest')
    #ax.set_title('dropped spines')
    # Move left and bottom spines outward by 10 points
    #ax.spines['left'].set_position(('outward', 10))
    #ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    import matplotlib.ticker as plticker

    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    labels0 = [' '] + s0
    labels1 = [' '] + s1
    ax.set_yticklabels(labels0)
    #ax.set_xticklabels()
    ax.set_xticklabels(labels1, rotation=90)

    ax.yaxis.set_tick_params(labelsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    #for tick in ax.xaxis.get_major_ticks():
    #    tick.label.set_rotation('vertical')

    for y, x in alignment:
        rect = patches.Rectangle((x - 0.5,y - 0.5),1,1,linewidth=4,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.figure(figsize=(1,1))
    plt.show()
    #fig.savefig("q_matrix.pdf", bbox_inches='tight')

def lcs_mat(m, d=1, not_connect=[], base=(0, 0)):
    m0, m1 = m.shape
    lengths = [[0 for j in range(m1+1)] for i in range(m0+1)]
    #print len(lengths), len(lengths[0])
    # row 0 and column 0 are initialized to 0 already
    for i in range(m0):
        lengths[i+1][0] = lengths[i][0] - d
    for j in range(m1):
        lengths[0][j+1] = lengths[0][j] - d
    for i in range(m0):
        for j in range(m1):
            # to skip i+base[0], j+base[1]
            if (i+base[0], j+base[1]) not in not_connect:
                lengths[i+1][j+1] = max(lengths[i+1][j]-d, lengths[i][j+1]-d, lengths[i][j]+m[i][j])
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j]-d, lengths[i][j+1]-d)
    return lengths

def align(lengths, d, s0, s1):
    if type(s0) is not list:
        s0 = s0.split(' ')
    if type(s1) is not list:
        s1 = s1.split(' ')
    result = []
    x, y = len(s0), len(s1)
    result.append((x, y))
    commons = [[], []]
    while x != 0 and y != 0:
        #if lengths[x][y] == lengths[x-1][y] - d:
        if abs(lengths[x][y] - (lengths[x-1][y] - d)) < 0.0001:
            x -= 1
        #elif lengths[x][y] == lengths[x][y-1] - d:
        elif abs(lengths[x][y] - (lengths[x][y-1] - d)) < 0.0001:
            y -= 1
        else:
            #assert a[x-1] == b[y-1]
            #result = a[x-1] + result
            #result.append((s0[x-1], s1[y-1]))
            #print x-1, y-1
            commons[0] = [x-1] + commons[0]
            commons[1] = [y-1] + commons[1]
            x -= 1
            y -= 1
        result.append((x, y))
    return result, commons


'''
def msa(lm_model, corpus, s0, s1, not_connect=[], base=(0, 0), gap_p=0.0, minus=0.0):
    # gap_p: gap penalty
    # minus: minus threshold
    m = get_matrix(lm_model, corpus, s0, s1)
    if minus > 0:
        m = m - minus
    length = lcs_mat(m, gap_p, not_connect, base)
    alignment, commons = align(length, gap_p, s0, s1)
    return alignment, commons
'''



def sent_simi(h0, h1, s0, s1, not_connect=[], base=(0, 0), gap_p=0.0, minus=0.0):
    # gap_p: gap penalty
    # minus: minus threshold
    m = get_matrix_from_hidden(h0, h1)
    if minus > 0:
        m = m - minus
    length = lcs_mat(m, gap_p, not_connect, base)
    return length[-1][-1]
    #alignment, commons = align(length, gap_p, s0, s1)
    #return alignment, commons


def msa(h0, h1, s0, s1, not_connect=[], base=(0, 0), gap_p=0.0, minus=0.0):
    # gap_p: gap penalty
    # minus: minus threshold
    m = get_matrix_from_hidden(h0, h1)
    if minus > 0:
        m = m - minus
    length = lcs_mat(m, gap_p, not_connect, base)
    alignment, commons = align(length, gap_p, s0, s1)
    return alignment, commons

