import lattice
import json
import argparse
from IPython import embed
import networkx as nx


def lattice_to_latex(tokens, G_lattice):
    G = lattice.init_graph(tokens)
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

    G = nx.node_link_graph(G_lattice)

    for n in G.nodes():
        text += '(%d) ' % (n)
        for a, b in G.out_edges(n):
            if len(G[a][b]['word']) == 0:
                word = ''
            else:
                word = '/'.join(G[a][b]['word'])
            text += 'edge node [above, sloped] {$%s$} (%d)\n' % (word, b)
    text += ''';
      \\end{tikzpicture}
    \\end{document}
    '''
    return text


parser = argparse.ArgumentParser(description='Print Lattice into Latex')
parser.add_argument('-original_dataset', default='data/dataset_small.json', help='path of the original dataset')
parser.add_argument('-lattice_dataset', default='data/dataset_soft_soft_0.60.json', help='path of the lattice json file')
parser.add_argument('-lattice_index', default=1, type=int, help='index of the lattice converted to latex')

if __name__ == '__main__':

    opt = parser.parse_args()

    with open(opt.original_dataset, 'r') as f:
        refs = json.load(f)
    with open(opt.lattice_dataset, 'r') as f:
        pseudo_refs = json.load(f)

    ref = refs['images'][opt.lattice_index]
    pseudo_ref = pseudo_refs['images'][opt.lattice_index]
    if ref['split'] == 'test' or ref['split'] == 'val':
        print "This example is in test set or validation set. Haven't convert to lattice"
    else:
        sentences = ref['sentences']
        tokens = [s['tokens'] for s in sentences]
        print lattice_to_latex(tokens, pseudo_ref['lattice'])
