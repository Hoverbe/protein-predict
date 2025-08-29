from argparse import ArgumentParser
# from utils import *
import pandas as pd
import networkx as nx
import random
import os

from utils import *
parser = ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True)
args = parser.parse_args()
source = args.mode
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pos_fname = ""
if source == 'neg':
    pos_fname = os.path.join(project_root, 'data','raw','negative_seqs_v3.fasta')
    pos_blast = pd.read_csv(os.path.join(project_root, 'data','neg_seqs_v3_self_blast.tsv'),
                            sep='\t', comment='#',  #
                            names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
                                   'tstart',
                                   'tend', 'e_value', 'bits'])
elif source == 'pos':
    pos_fname = os.path.join(project_root,'data', 'five-folds-data','positive_seqs_v3.fasta')
    pos_blast = pd.read_csv(os.path.join(project_root, 'data','pos_seqs_v3_self_blast.tsv'),
                            sep='\t', comment='#',  #
                            names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
                                   'tstart',
                                   'tend', 'e_value', 'bits'])
pos_h, pos_s = ReadFastaFile(pos_fname)
h2s = {h: s for h, s in zip(pos_h, pos_s)}
s2h = {s: h for h, s in zip(pos_h, pos_s)}

seq_graph = nx.Graph()
edges = [(t[1], t[2]) for t in pos_blast.itertuples() if t[3] >= 90 and t[1] != t[2]]
seq_graph.add_edges_from(edges)
nodes = list(seq_graph.nodes.keys())
count = 0
for h in pos_h:
    if h.split()[0] not in nodes:
        count=count+1
        seq_graph.add_node(h.split()[0])
print(count)
components = list(nx.connected_components(seq_graph))
components.sort(key=lambda x: len(x), reverse=True)
remove_hs = []
for clu in components:
    if len(clu) > 1:
        keep_h = random.choice(list(clu))
        remove_hs.extend(list(clu - {keep_h}))
assert len(remove_hs) + len(components) == len(pos_h)
pos_h = list(set(pos_h) - set(remove_hs))
pos_s = [h2s[h] for h in pos_h]
print(len(pos_h))
if source == 'neg':
    SaveFastaFile(os.path.join(project_root,'data','negative_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta') ,pos_h, pos_s)
elif source == 'pos':
    SaveFastaFile(os.path.join(project_root,'data','positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta') ,pos_h, pos_s)