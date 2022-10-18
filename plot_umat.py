import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import pickle
import functools
import scipy.sparse.csgraph as csgraph
import som_seq
import jax_imports
import minsptree as mspt
import quicksom.som
import quicksom.somax
import quicksom.utils
from Timer import Timer


timer = Timer(autoreset=True)

def main(somfile,outname='umat',delimiter=None,hideSeqs=False,minsptree=False,unfold=False):

    #Load the data and parse it
    with open(somfile, 'rb') as somfileaux:
        som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
    bmus = list(zip(*som.bmus.T))
    titles = som.labels
    #bmus = list(zip(*som.bmus[0:100].T))
    #titles = som.labels[0:100]
    titles = [title.replace(">","") for title in titles]
    if delimiter != None:
        labels = [title.split(delimiter)[0] for title in titles]

    #Compute the local Adjacency Matrix between the qbmus
    if minsptree or unfold:
        timer.start('computing localadj between queries')
        localadj, localadj_paths = mspt.get_localadjmat(som.umat,som.adj,bmus,verbose=True)
        timer.stop()

    if unfold:
        #Compute the minimal spanning tree
        timer.start('compute the msptree')
        msptree = csgraph.minimum_spanning_tree(localadj)
        timer.stop()
        #Use the minimial spanning three between queries bmus to unfold the umat
        timer.start('compute the umap unfolding')
        uumat,mapping,reversed_mapping = mspt.get_unfold_umat(som.umat, som.adj, bmus, msptree)
        timer.stop()
        som.uumat = uumat
        som.mapping = mapping
        som.reversed_mapping = reversed_mapping
        auxumat = uumat
        unfbmus = [mapping[bmu] for bmu in bmus]
        auxbmus = unfbmus
        timer.start('compute the unfolded adj matrix')
        som._get_unfold_adj()
        auxadj = som.uadj
    else:
        auxbmus = bmus
        auxumat = som.umat
        auxadj = som.adj

    n1, n2 = auxumat.shape

    #Compute the msptree pairs and paths between the qbmus
    if minsptree:
        timer.start('get the minsptree paths')
        msptree_pairs, msptree_paths = mspt.get_minsptree(localadj,localadj_paths)
        timer.stop()
        if unfold:
            _n1,_n2 = som.umat.shape
            unf_msptree_pairs = []
            unf_rpaths = {}
            for pair in msptree_pairs:
                unf_rpair = [mspt.get_uumat_ravel_cell(pair[0],(_n1,_n2),(n1,n2),mapping), mspt.get_uumat_ravel_cell(pair[1],(_n1,_n2),(n1,n2),mapping)]
                unf_msptree_pairs.append(unf_rpair)
            unf_msptree_pairs = np.asarray(unf_msptree_pairs)
            msptree_pairs = unf_msptree_pairs
            for k in msptree_paths:
                unf_rk = (mspt.get_uumat_ravel_cell(k[0],(_n1,_n2),(n1,n2),mapping),mspt.get_uumat_ravel_cell(k[1],(_n1,_n2),(n1,n2),mapping))
                unf_rpath = [mspt.get_uumat_ravel_cell(step,(_n1,_n2),(n1,n2),mapping) for step in msptree_paths[k]]
                unf_rpaths[unf_rk] = unf_rpath
            msptree_paths = unf_rpaths

    _plot_umat(auxumat,auxbmus,labels,hideSeqs)

    if minsptree:
        _plot_msptree(msptree_pairs, msptree_paths,(n1,n2))

    plt.savefig(outname+'.pdf')
    plt.show()

def _plot_msptree(msptree_pairs, msptree_paths,somsize,verbose=False):
    n1, n2 = somsize
    for i,msptree_pair in enumerate(msptree_pairs):
        if verbose: print('Printing the shortest parth between %s and %s'%(msptree_pair[0],msptree_pair[1]))
        msptree_path = msptree_paths[tuple(msptree_pair)]
        _msptree_path = np.asarray(np.unravel_index(msptree_path, (n1, n2)))
        _msptree_path = np.vstack((_msptree_path[0], _msptree_path[1])).T
        for j,step in enumerate(_msptree_path):
            if j == 0: continue
            #Check to avoid borders printting horizontal or vertical lines
            if (_msptree_path[j-1][0] == 0 and _msptree_path[j][0] == n1-1) or (_msptree_path[j-1][0] == n1-1 and _msptree_path[j][0] == 0) or (_msptree_path[j-1][1] == 0 and _msptree_path[j][1] == n2-1) or (_msptree_path[j-1][1] == n2-1 and _msptree_path[j][1] == 0): continue
            aux = np.stack((_msptree_path[j-1],_msptree_path[j])).T
            plt.plot(aux[1], aux[0],c='w',linewidth=0.8)


def _plot_umat(umat, bmus, labels, hideSeqs):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(umat)
    figure.colorbar(cax)

    if not hideSeqs:
        if len(labels) == 0:
            for bmu in bmus:
                mspt.highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=0.5)
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])
            for unique_label in list(set(labels)):
                aux_X = []
                aux_Y = []
                for i,label in enumerate(labels):
                    if label == unique_label:
                        aux_X.append(bmus[i][1])
                        aux_Y.append(bmus[i][0])
                    else: continue
                ax.scatter(aux_X,aux_Y,label=unique_label,s=15)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                              fancybox=True, shadow=True, ncol=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-o', '--outname', help = 'Output name for the umat',default='umat')
    parser.add_argument('-d', '--delimiter', help = 'If gruping infomation was contained in the sequences title, the delimiter split it and select the prefix',default=None)
    parser.add_argument('--hide_seqs',help = 'To hide input sequences',action='store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between BMUs', default = False, action = 'store_true')
    parser.add_argument('--unfold',help='Unfold the Umat', default = False, action = 'store_true')
    args = parser.parse_args()

    main(somfile=args.som,outname=args.outname,delimiter=args.delimiter,hideSeqs=args.hide_seqs,minsptree=args.minsptree, unfold=args.unfold)
