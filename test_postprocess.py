import minsptree as msptree
import quicksom.som
import quicksom.somax
import quicksom.utils
import functools
import som_seq
from som_seq import seqmetric
import pickle
from Timer import Timer
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
import newick
import numpy as np
import networkx as nx

somfile = 'som.pickle'
timer = Timer(autoreset=True)

with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
som.metric = functools.partial(som_seq.seqmetric, b62=b62)
bmus = list(zip(*som.bmus[0:100].T))
timer.start('computing localadj between queries')
localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus)
timer.stop()
timer.start('computing the tree')
mstree = csgraph.minimum_spanning_tree(localadj)
mstree = mstree.tocoo()
mstree_nodes = np.concatenate((mstree.row,mstree.col))
mstree_nodes = list(set(list(mstree_nodes)))
mstree_nodes_labels = quicksom.utils.bmus_to_label(mstree_nodes,som.bmus,som.labels,(som.m,som.n))
mstree_nodes_labels = [';'.join(node_label).replace(">","") for node_label in mstree_nodes_labels]
mapping = dict(zip(mstree_nodes,mstree_nodes_labels))

mstree_ntw = nx.from_scipy_sparse_matrix(mstree)
mstree_ntw_isolates = list(nx.isolates(mstree_ntw))
mstree_ntw.remove_nodes_from(mstree_ntw_isolates)
print(mstree_ntw)
nx.relabel_nodes(mstree_ntw,mapping,copy=False)
nx.write_gml(mstree_ntw,'mstree.gml')


timer.stop()
timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = msptree.get_unfold_umat(som.umat, som.adj,bmus, mstree)
timer.stop()

unfbmus = [mapping[bmu] for bmu in bmus]

plt.matshow(uumat)
plt.colorbar()
plt.savefig('testunfold.pdf')
