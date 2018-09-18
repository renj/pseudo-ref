#from language_model import loader
import msa_class
import torch
import numpy as np
import networkx as nx
from torch.autograd import Variable
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import json
import pickle
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
#from IPython import embed
from multiprocessing import Pool
import random
import time
import argparse


def lattice_to_latex(tokens, lattice):
    G = init_graph(tokens)
    text = '''\\documentclass[tikz,border=10pt]{standalone}
    \\usetikzlibrary{automata,positioning,arrows.meta}
    \\begin{document}
      \\begin{tikzpicture}
        [
          initial/.style={line width=1pt},
          accepting by double/.append style={line width=1pt},
          semithick,
        ]\n'''
    text += '\\node (0) [state, initial] {$0$};\n'
    k_list = sorted(G[0].keys())
    n_sents = len(G[0].keys())

    text += '\\node (%d) [state, above right=of 0] {$%d$};\n' % (k_list[1], k_list[1])
    text += '\\node (%d) [state, right=of 0, above=of %d] {$%d$};\n' % (k_list[0], k_list[1], k_list[0])
    if n_sents > 2:
        text += '\\node (%d) [state, right=of 0] {$%d$};\n' % (k_list[2], k_list[2])
    if n_sents > 3:
        text += '\\node (%d) [state, below right=of 0] {$%d$};\n' % (k_list[3], k_list[3])
    if n_sents > 4:
        text += '\\node (%d) [state, right=of 0, below=of %d] {$%d$};\n' % (k_list[4], k_list[3], k_list[4])

    for n in G.nodes():
        if n == -1 or n in k_list or n == 0:
            continue
        text += '\\node (%d) [state, ' % (n)
        for a, b in G.in_edges(n):
            text += 'right=of %d] {$%d$};' % (a, n)
            break
        text += '\n'
    n = -1
    text += '\\node (%d) [state, ' % (n)
    for a, b in G.in_edges(n):
        text += 'right=of %d,' % (a)
    text += '] {$%d$};\n' % (n)
    text += '\\path [-{Stealth[]}]\n\n'

    G = lattice
    thr = get_threshold(G)

    for n in G.nodes():
        text += '(%d) ' % (n)
        for a, b in G.out_edges(n):
            if len(G[a][b]['word']) == 0:
                word = ''
            else:
                word = '/'.join(G[a][b]['word'])
            text += 'edge node [above, sloped] {$%s$} (%d)\n' % (word, b)
            text += 'edge node [below, sloped] {$%d$} (%d)\n' % (thr[a][b], b)
    text += ''';
      \\end{tikzpicture}
    \\end{document}
    '''
    return text


def lcs(a, b, not_connect=[], base=(0,0)):
    #a, b = sorted([a, b])
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y and (i+base[0], j+base[1]) not in not_connect:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    result = []
    x, y = len(a), len(b)
    commons = [[], []]
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            #result = a[x-1] + result
            result.append(a[x-1])
            #print x-1, y-1
            commons[0] = [x-1] + commons[0]
            commons[1] = [y-1] + commons[1]
            x -= 1
            y -= 1
    return result, commons, lengths

def init_graph(sentences):
    G = nx.DiGraph()
    idx = 1
    for i, line in enumerate(sentences):
        for j, w in enumerate(line):
            if j == 0:
                G.add_edge(0, idx, word=[], idx=[])
            else:
                G.add_edge(idx, idx+1, word=[], idx=[])
                idx += 1
            G.add_edge(idx, idx+1, word=[w], idx=[(i,j)])
            idx += 1
            if j == len(sentences[i]) - 1:
                #end
                G.add_edge(idx, -1, word=[], idx=[])
                idx += 1
        #break
    return G

def get_node(G, idx):
    for u, v in G.edges():
        if idx in G[u][v]['idx']:
            return u, v

def merge_edges(G, (a0, a1), (b0, b1)):
    # merge edge b to a
    # if a == b: do nothing
    if b0 == a0 and b1 == a1:
        return

    if b0 != a0:
        G.add_edge(b0, a0, word=[], idx=[])
    if a1 != b1:
        G.add_edge(a1, b1, word=[], idx=[])
    G[a0][a1]['idx'] += G[b0][b1]['idx']
    for w in G[b0][b1]['word']:
        if w not in G[a0][a1]['word']:
            G[a0][a1]['word'].append(w)
    G.remove_edge(b0, b1)

def merge(G, commons, sent_a, sent_b):
    for idx_a, idx_b in zip(commons[0], commons[1]):
        #print idx_a, idx_b
        a0, a1 = get_node(G, idx=(sent_a, idx_a))
        b0, b1 = get_node(G, idx=(sent_b, idx_b))
        #print a0, a1, ';', b0, b1
        merge_edges(G, (a0, a1), (b0, b1))

def get_threshold(G):
    thr = defaultdict(dict)
    queue = list(G.edges())
    while len(queue) != 0:
        v, u = queue.pop(0)
        if len(G[v][u]['idx']) != 0:
            #thr[v][u] = len(G[v][u]['idx'])
            cnt = defaultdict(int)
            for i_sent, i_word in G[v][u]['idx']:
                cnt[i_sent] += 1
            thr[v][u] = max(cnt.values())
            continue
        tmp_a = [thr[a][b] for a, b in G.out_edges(u) if a in thr and b in thr[a]]
        tmp_b = [thr[a][b] for a, b in G.in_edges(v) if a in thr and b in thr[a]]
        if len(tmp_a) == 0 and len(tmp_b) == 0:
            queue.append((v, u))
            continue
        tmp_a = 0 if len(tmp_a) == 0 else max(tmp_a)
        tmp_b = 0 if len(tmp_b) == 0 else max(tmp_b)
        threshold = max(tmp_a, tmp_b)
        thr[v][u] = threshold
    return thr


def dfs(G, T, v=0, visited=defaultdict(int)):
    if v == -1:
        ret = [[]]
        return ret
    else:
        ret = []
    for u in G[v]:
        threshold = T[v][u]
        #if visited[(v,u)] >= 1:
        if visited[(v,u)] >= threshold:
            continue
        visited[(v,u)] += 1

        for line in dfs(G, T, u, visited):
            if line == []:
                line = ''
            if len(G[v][u]['word']) == 0:
                _sent = line
                ret.append(_sent)
            else:
                for w in G[v][u]['word']:
                    if line == '':
                        _sent = w
                    else:
                        _sent = w + ' ' + line
                    ret.append(_sent)
                    #break
        visited[(v,u)] -= 1
    if len(ret) == 0:
        return []
    if ret != [[]]:
        ret = list(set(ret))
    return ret



def check_cycle(G):
    try:
        nx.find_cycle(G)
    except Exception:
        return False
    return True


def acyclic_lm_commons(G, sentences, hiddens, row, col, minus=0.0):
    # row: sentence a
    # col: sentence b
    row_idx = {}
    col_idx = {}
    for u, v in G.edges():
        for pair in [p for p in G[u][v]['idx'] if p[0] == row]:
            if u in row_idx or v in row_idx:
                print 'Warning'
            row_idx[u] = pair[1]
            row_idx[v] = pair[1]
        for pair in [p for p in G[u][v]['idx'] if p[0] == col]:
            if u in col_idx or v in col_idx:
                print 'Warning'
            col_idx[u] = pair[1]
            col_idx[v] = pair[1]
    ret = nx.floyd_warshall(G)
    not_connect = set()
    for r in row_idx:
        for c in col_idx:
            if ret[r][c] < float('inf') or ret[c][r] < float('inf'):
                not_connect.add((row_idx[r],col_idx[c]))
    do_connect = set()
    for idx in set(col_idx.keys()) & set(row_idx.keys()):
        do_connect.add((row_idx[idx], col_idx[idx]))
    do_connect.add((len(sentences[row]), len(sentences[col])))
    left_pair = (0, 0)
    ranges = []
    for a, b in sorted(list(do_connect)):
        now = ((left_pair[0], a), (left_pair[1], b))
        ranges.append(now)
        left_pair = (a+1, b+1)
    commons = [[], []]
    for a, b in ranges:
        _alignment, _commons = msa_class.msa(hiddens[row][a[0]:a[1]], hiddens[col][b[0]:b[1]], sentences[row][a[0]:a[1]], sentences[col][b[0]:b[1]], not_connect, base=(a[0], b[0]), minus=minus)

        commons[0] += [_i + a[0] for _i in _commons[0]]
        commons[1] += [_i + b[0] for _i in _commons[1]]
        if a[1] < len(sentences[row]) and b[1] < len(sentences[col]):
            commons[0] += [a[1]]
            commons[1] += [b[1]]

    for a, b in zip(commons[0], commons[1]):
        if (a, b) in not_connect and (a, b) not in do_connect:
            print 'Warning', (a,b)

    return commons

def acyclic_commons(G, sentences, row, col):
    # row: sentence a
    # col: sentence b
    '''
    merged_edge = []
    for u, v in G.edges():
        i_list = [i_sent for i_sent, i_word in G[u][v]['idx']]
        if row in i_list and col in i_list:
            merged_edge.append((u,v))
    if len(merged_edge) != 0:
        #print merged_edge
        z = 0
    else:
        result, commons, L = lcs(sentences[row], sentences[col])
        merge(G, commons, row, col)
    '''
    row_idx = {}
    col_idx = {}
    for u, v in G.edges():
        for pair in [p for p in G[u][v]['idx'] if p[0] == row]:
            #row_idx.append((u, pair[1]))
            #row_idx.append((v, pair[1]))
            if u in row_idx or v in row_idx:
                print 'Warning'
            row_idx[u] = pair[1]
            row_idx[v] = pair[1]
        for pair in [p for p in G[u][v]['idx'] if p[0] == col]:
            if u in col_idx or v in col_idx:
                print 'Warning'
            col_idx[u] = pair[1]
            col_idx[v] = pair[1]
    ret = nx.floyd_warshall(G)
    not_connect = set()
    for r in row_idx:
        for c in col_idx:
            if ret[r][c] < float('inf') or ret[c][r] < float('inf'):
                #print r,c,ret[r][c]
                not_connect.add((row_idx[r],col_idx[c]))
            '''
            if ret[c][r] < float('inf'):
                not_connect.add((col_idx[c], row_idx[r]))
            '''
    do_connect = set()
    for idx in set(col_idx.keys()) & set(row_idx.keys()):
        do_connect.add((row_idx[idx], col_idx[idx]))
    do_connect.add((len(sentences[row]), len(sentences[col])))
    left_pair = (0, 0)
    ranges = []
    for a, b in sorted(list(do_connect)):
        now = ((left_pair[0], a), (left_pair[1], b))
        ranges.append(now)
        left_pair = (a+1, b+1)
    commons = [[], []]
    #not_connect = set()
    for a, b in ranges:
        _result, _commons, _L = lcs(sentences[row][a[0]:a[1]], sentences[col][b[0]:b[1]], not_connect, base=(a[0], b[0]))
        commons[0] += [_i + a[0] for _i in _commons[0]]
        commons[1] += [_i + b[0] for _i in _commons[1]]
        if a[1] < len(sentences[row]) and b[1] < len(sentences[col]):
            commons[0] += [a[1]]
            commons[1] += [b[1]]

    #print not_connect

    for a, b in zip(commons[0], commons[1]):
        if (a, b) in not_connect and (a, b) not in do_connect:
            print 'Warning', (a,b)

    return commons


def generate_lattice(sentences, hiddens, order_method, align_method, get_G=False, lm=False, minus=0.0, simi_mat=None):
    G = init_graph(sentences)

    if order_method == 'hard':
        vectorizer = CountVectorizer()    
        matrix = vectorizer.fit_transform([' '.join(line) for line in sentences])
        simi = cosine_similarity(matrix)
    elif order_method == 'soft':
        simi_mat = load_simi_mat_from_hiddens(hiddens, minus)
        simi = np.array(simi_mat)

    if align_method == 'hard':
        lm = False
    elif align_method == 'soft':
        lm = True

    visited = set()
    if order_method == 'random':
        seq = [(i, j) for i in range(len(sentences)) for j in range(len(sentences))]
        random.shuffle(seq)
    else:
        seq = simi.reshape(1, -1).argsort()[0][::-1]
    for i in seq:
        if order_method == 'random':
            row, col = i
        else:
            row = i % simi.shape[0]
            col = int(math.floor(i / simi.shape[0]))
        row, col = sorted([row, col])
        if row == col:
            continue
        if row in visited and col in visited:
            continue
        if (col, row) in visited or (row, col) in visited:
            continue
        if order_method != 'random' and simi[row][col] < minus:
            break
        if lm:
            commons = acyclic_lm_commons(G, sentences, hiddens, row, col,  minus)
            merge(G, commons, row, col)
            if check_cycle(G):
                print 'cycle exist'
                return G
        else:
            result, commons, L = lcs(sentences[row], sentences[col])
            merge(G, commons, row, col)

        visited.add(col)
        visited.add(row)
        visited.add((col, row))

        '''
        ret = dfs(G, 0)
        ret = set([' '.join(line) for line in ret])
        refs.append(ret)
        print(len(ret))
        '''

    n_edges = len(G.edges())

    while True:
        G = simplify(G)
        n_edges_now = len(G.edges())
        if n_edges == n_edges_now:
            break
        n_edges = n_edges_now

    if get_G:
        return G

    T = get_threshold(G)
    ret = dfs(G, T)
    #ret = set([' '.join(line) for line in ret])
    ret = set(ret)
    #print ret

    #return ret, refs, G
    return ret

def simplify(G):
    for n in G.nodes():
        if len(G.in_edges(n)) == 1 and len(G.out_edges(n)) == 1:
            a, b = list(G.in_edges(n))[0]
            u, v = list(G.out_edges(n))[0]
            if len(G[a][b]['word']) == 0 and len(G[u][v]['word']) == 0:
                if (a, v) in G.edges() and len(G[a][v]['word']) != 0:
                    continue
                #print a, b, u, v, G[a][b], G[u][v]
                G.remove_edge(a, b)
                G.remove_edge(u, v)
                if a == v:
                    continue
                G.add_edge(a, v, word=[], idx=[])
    return G


def generate_refs(sentences, order_method, align_method, minus, sentid, simi_mat=None):
    tokens = [s['tokens'] for s in sentences]
    if align_method == 'soft':
        hiddens = [s['hidden'] for s in sentences]
    else:
        hiddens = None

    refs = generate_lattice(tokens, hiddens, order_method, align_method, simi_mat=simi_mat, minus=minus)

    for e in tokens:
        refs.add(' '.join(e))
    refs = list(refs)
    bleu_scorer = BleuScorer(n=4)
    for ref in refs:
        bleu_scorer += (ref, [' '.join(e) for e in tokens])
    score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
    new_sentences = []
    for i, s in enumerate(scores[3]):
        new_ref = {}
        new_ref['imgid'] = sentences[0]['imgid']
        new_ref['raw'] = refs[i]
        new_ref['tokens'] = refs[i].split(' ')
        new_ref['sentid'] = sentid
        new_ref['bleu'] = s
        new_sentences.append(new_ref)
        sentid += 1

    return new_sentences
 

def load_hiddens(j, model, dictionary, batch_size=128):
    outputs = []
    crt_size = 0
    txts = []
    t0 = time.time()
    mat = []
    idx = 0
    for img in j['images']:
        idx += 1
        for sent in img['sentences']:
            #hiddens.append(h)
            txts.append(sent['tokens'])
            crt_size += 1
            if crt_size >= batch_size:
                t_len = [len(txt) for txt in txts]
                max_len = max(t_len)
                for txt in txts:
                    new_txt = [dictionary[w] for w in txt] + [0] * (max_len - len(txt))
                    mat.append(new_txt)
                feature = torch.from_numpy(np.array(mat))
                feature = Variable(feature.t().cuda(model.device))
                emb = model.encoder(feature)
                #print '=== emb device ===', model.device, torch.backends.cudnn.version()
                output, hidden = model.rnn(emb)
                
                for i, txt in enumerate(txts):
                    print i, txt,
                    outputs.append(output[:len(txt), i, :].cpu().data.numpy())
                    #outputs.append(output[:len(txt), i, :].cpu().data)
                    
                mat = []
                crt_size = 0
                txts = []
                #flag = True
                #break
        #if flag:
        #    break
        if crt_size > 0:
            t_len = [len(txt) for txt in txts]
            max_len = max(t_len)
            for txt in txts:
                new_txt = [dictionary[w] for w in txt] + [0] * (max_len - len(txt))
                mat.append(new_txt)
            #embed()
            feature = torch.from_numpy(np.array(mat))
            feature = Variable(feature.t().cuda(model.device))
            emb = model.encoder(feature)
            output, hidden = model.rnn(emb)
            
            for i, txt in enumerate(txts):
                outputs.append(output[:len(txt), i, :].cpu().data.numpy())
                #outputs.append(output[:len(txt), i, :].cpu().data)
                
            mat = []
            crt_size = 0
            txts = []
    
    for i, txt in enumerate(txts):
        outputs.append(output[:len(txt), i, :].cpu().data.numpy())

    idx = 0
    for img in j['images']:
        for sent in img['sentences']:
            sent['hidden'] = outputs[idx]
            idx += 1

    t1 = time.time()
    print 'Load hidden completed!', t1 - t0

def load_simi_mat(img, minus):
    sents = img['sentences']
    simi_mat = [[0 for _ in sents] for _ in sents]
    for idx in range(len(sents)):
        for jdx in range(idx+1, len(sents)):
            h0 = sents[idx]['hidden']
            h1 = sents[jdx]['hidden']
            s0 = sents[idx]['tokens']
            s1 = sents[jdx]['tokens']
            simi = msa_class.sent_simi(h0, h1, s0, s1, minus=minus)
            simi_mat[idx][jdx] = simi
            simi_mat[jdx][idx] = simi
    #img['simi_mat'] = simi_mat
    return simi_mat

def load_simi_mat_from_hiddens(hiddens, minus, imgid=0):
    simi_mat = [[0 for _ in hiddens] for _ in hiddens]
    for idx in range(len(hiddens)):
        for jdx in range(idx+1, len(hiddens)):
            h0 = hiddens[idx]
            h1 = hiddens[jdx]
            simi = msa_class.sent_simi(h0, h1, '', '', minus=minus)
            simi_mat[idx][jdx] = simi
            simi_mat[jdx][idx] = simi
    return simi_mat


parser = argparse.ArgumentParser(description='Pseudo-Ref Generation.')
parser.add_argument('-order_method', default='hard', help='Method of ordering. Options are [hard|soft|random]')
parser.add_argument('-align_method', default='hard', help='Method of alignment. Options are [hard|soft]')
parser.add_argument('-minus', default=0.6, type=float, help='Minimum threshold')
parser.add_argument('-gpuid', default=1, type=int, help='GPUid')
parser.add_argument('-save_graph', action="store_true", help='save lattice in new_j')
parser.add_argument('-multi_process', action="store_true", help='enable multi processing')
parser.add_argument('-n_cpu', default=25, type=int, help='number of threads')
parser.add_argument('-dataset', default='data/dataset_small.json', help='number of threads')
parser.add_argument('-lm_dictionary', default='data/LM_coco.dict', help='dictionary file of language model')
parser.add_argument('-lm_model', default='data/LM_coco.pth', help='number of threads')


if __name__ == '__main__':

    opt = parser.parse_args()

    with open(opt.dataset, 'r') as f:
        j = json.load(f)
    with open(opt.dataset, 'r') as f:
        new_j = json.load(f)

    print('========= READ DATA COMPLETE ===========')

    file_name = 'data/dataset_%s_%s_%.2f.json'%(opt.order_method, opt.align_method, opt.minus)

    if opt.align_method == 'hard':
        lm_model = None
        dictionary = None
    elif opt.align_method == 'soft':
        # Load Language Model for soft alignment, 
        #batch_size = 20
        #text_field = data.Field(lower=True)
        #label_field = data.Field(lower=True)
        #corpus = loader.Corpus(opt.lm_data, text_field, label_field, batch_size)
        #dictionary = text_field.vocab.stoi
        with open(opt.lm_dictionary, 'r') as f:
            dictionary = pickle.load(f)


        #with open('../pytorch-examples/word_language_model/LM_coco_60_0_BiLangModel.pth') as f:
        with open(opt.lm_model) as f:
            lm_model = torch.load(f, map_location=lambda storage, loc: storage)
            #print '=== LM device ===', lm_model.device, torch.backends.cudnn.version()
            lm_model.cuda(opt.gpuid)
            lm_model.device = opt.gpuid
        load_hiddens(j, lm_model, dictionary)
    else:
        print 'wrong align method'
        exit()

    print 'Save to file:', file_name


    if opt.multi_process:
        pool = Pool(opt.n_cpu)
        n_left = 0
        sentences_pool = []
        idxs = []

    t0 = time.time()
    sentid = 0
    for idx in range(len(j['images'])):
        if j['images'][idx]['split'] == 'test' or j['images'][idx]['split'] == 'val':
            # This example is in test set or validation set
            continue
        sentences = j['images'][idx]['sentences']
        #sentences = [s for s in sentences if len(s['tokens']) <= 20]

        simi_mat = None
 
        if opt.multi_process:
            if opt.save_graph:
                tokens = [s['tokens'] for s in sentences]
                if opt.align_method == 'soft':
                    hiddens = [s['hidden'] for s in sentences]
                else:
                    hiddens = None
                sentences_pool.append(pool.apply_async(generate_lattice, (tokens, hiddens, opt.order_method, opt.align_method, True, False, opt.minus, simi_mat,)))

            else:
                sentences_pool.append(pool.apply_async(generate_refs, (sentences, opt.order_method, opt.align_method, opt.minus, sentid, simi_mat)))
            idxs.append(idx)
            if n_left < opt.n_cpu:
                n_left += 1
                continue
            else:
                for idx, new_sentences in zip(idxs, [p.get(99999999) for p in sentences_pool]):
                    if opt.save_graph:
                        G = new_sentences
                        new_j['images'][idx]['lattice'] = nx.node_link_data(G)
                        t1 = time.time()
                        print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0)
                    else:
                        sentid += len(new_sentences)
                        bleus = [s['bleu'] for s in new_sentences]
                        new_j['images'][idx]['sentences'] = new_sentences
                        t1 = time.time()
                        print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0), '#refs:', len(new_sentences)
                    t0 = t1

                n_left = 0
                sentences_pool = []
                idxs = []

        else:
            if opt.save_graph:
                tokens = [s['tokens'] for s in sentences]
                if opt.align_method == 'soft':
                    hiddens = [s['hidden'] for s in sentences]
                else:
                    hiddens = None
                G = generate_lattice(tokens, hiddens, opt.order_method, opt.align_method, minus=opt.minus, simi_mat=simi_mat, get_G=True)
                #new_j['images'][idx]['lattice'] = G
                new_j['images'][idx]['lattice'] = nx.node_link_data(G)
                t1 = time.time()
                print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0)
            else:
                new_sentences = generate_refs(sentences, opt.order_method, opt.align_method, opt.minus, sentid, simi_mat)
                sentid += len(new_sentences)
                bleus = [s['bleu'] for s in new_sentences]
                new_j['images'][idx]['sentences'] = new_sentences
                t1 = time.time()
                print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0), '#refs:', len(new_sentences)
            t0 = t1

    if opt.multi_process and len(sentences_pool) > 0 and len(idxs) > 0:
        for idx, new_sentences in zip(idxs, [p.get(99999999) for p in sentences_pool]):
            if opt.save_graph:
                G = new_sentences
                new_j['images'][idx]['lattice'] = nx.node_link_data(G)
                t1 = time.time()
                print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0)
            else:
                sentid += len(new_sentences)
                bleus = [s['bleu'] for s in new_sentences]
                new_j['images'][idx]['sentences'] = new_sentences
                t1 = time.time()
                print '%d/%d'%(idx, len(j['images'])), 'time:%.3f'%(t1 - t0), '#refs:', len(new_sentences)
            t0 = t1

    with open(file_name, 'w') as f:
        json.dump(new_j, f)
    #embed()
