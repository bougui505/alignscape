import quicksom.som
import quicksom.somax
import functools
import os
import sys
import numpy as np
import torch
import dill as pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text
import ast
from random import randint
sys.path.insert(1, '/work/ifilella/quicksom_seq')
import seqdataloader as seqdataloader
import som_seq
import jax_imports
import minsptree as msptree
import scipy.sparse
import scipy.sparse.csgraph as graph
from skimage.feature import peak_local_max
from sklearn.cluster import AgglomerativeClustering

def main(somfile,bmusfile,outname='reumat.pdf',allinp=False,unfold=False,minsptree=False,save=None,load=None,remap=False,clustering=False):

    #Load the data (allbmus, the queries, the som and the subtype dicc)
    allbmus = np.genfromtxt(bmusfile, dtype=str, skip_header=1)
    with open(somfile, 'rb') as somfileaux:
            som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)

    #Parse the queries and their corresponding bmus
    labels = list()
    bmus = list()

    #Parse the data
    labels = list()
    bmus = list()
    for k,bmu in enumerate(allbmus):
        bmus.append((int(bmu[0]),int(bmu[1])))
        labels.append(bmu[-1].replace(">",""))

    #Load or compute the localadj matrix between the qbmus
    if load and minsptree:
        try:
            localadj = msptree.load_localadjmat(load + '_localadj.npz')
        except:
            raise KeyError('%s_localadj.npz is missing or have a wrong name'%load)
        try:
            with open(load+'_paths.pkl', 'rb') as f:
                paths = pickle.load(f)
        except:
            raise KeyError('%s_paths.pkl is missing or have a wrong name'%load)
    elif not load and minsptree:
        localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus,verbose=True)
        if save is not None:
            scipy.sparse.save_npz(save + '_localadj.npz', localadj)
            with open(save + '_paths.pkl', 'wb') as f:
                pickle.dump(paths, f)


    if unfold:

        #Get the mininimal spanning tree of the localadj matrix
        mstree = graph.minimum_spanning_tree(localadj)

        #Use the minimial spanning three between bmus to unfold the umat
        uumat,mapping,reversed_mapping = msptree.get_unfold_umat(som.umat, som.adj, bmus, mstree)

        som.uumat = uumat
        som.mapping = mapping
        som.reversed_mapping = reversed_mapping
        auxumat = uumat
        unfbmus = [mapping[bmu] for bmu in bmus]
        auxbmus = unfbmus
        som._get_unfold_adj()
        auxadj = som.uadj
    else:
        auxbmus = bmus
        auxumat = som.umat
        auxadj = som.adj
    n1, n2 = auxumat.shape

    if clustering:
        local_min = peak_local_max(-auxumat, min_distance=6)
        n_local_min = local_min.shape[0]
        clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_local_min)
        all_to_all_dist = graph.shortest_path(auxadj, directed=False)
        try:
            labels = clusterer.fit_predict(all_to_all_dist)
        except ValueError as e:
            print(f'WARNING : The following error was catched : "{e}"\n'
                  f'The clusterer yields zero clusters on the data.'
                  ' You should train it more or gather more data')
            labels = np.zeros(n1 * n2)
        labels = labels.reshape((n1, n2))
        plt.matshow(labels)
    else:
        plt.matshow(auxumat)
        plt.colorbar()

    if minsptree:
        #Get the minimal spanning tree of the queries
        if unfold:
            if remap:
                mstree, mstree_pairs, paths = msptree.get_minsptree(localadj,paths)
                _n1,_n2 = som.umat.shape
                unf_mstree_pairs = []
                unf_rpaths = {}
                for pair in mstree_pairs:
                    unf_rpair = [msptree.get_uumat_ravel_cell(pair[0],(_n1,_n2),(n1,n2),mapping), msptree.get_uumat_ravel_cell(pair[1],(_n1,_n2),(n1,n2),mapping)]
                    unf_mstree_pairs.append(unf_rpair)
                unf_mstree_pairs = np.asarray(unf_mstree_pairs)
                mstree_pairs = unf_mstree_pairs
                for i,k in enumerate(paths):
                    unf_rk = (msptree.get_uumat_ravel_cell(k[0],(_n1,_n2),(n1,n2),mapping),msptree.get_uumat_ravel_cell(k[1],(_n1,_n2),(n1,n2),mapping))
                    unf_rpath = [msptree.get_uumat_ravel_cell(step,(_n1,_n2),(n1,n2),mapping) for step in paths[k]]
                    unf_rpaths[unf_rk] = unf_rpath
                paths = unf_rpaths
            else:
                ulocaladj, upaths = msptree.get_localadjmat(auxumat,auxadj,auxbmus,verbose=True)
                mstree, mstree_pairs, paths = msptree.get_minsptree(ulocaladj,upaths)
        else:
             mstree, mstree_pairs, paths = msptree.get_minsptree(localadj,paths)

        #Print the minimal smapnning tree
        for i,mstree_pair in enumerate(mstree_pairs):
            print('Printing the shortest parth between %s and %s'%(mstree_pair[0],mstree_pair[1]))
            mstree_path = paths[tuple(mstree_pair)]
            _mstree_path = np.asarray(np.unravel_index(mstree_path, (n1, n2)))
            _mstree_path = np.vstack((_mstree_path[0], _mstree_path[1])).T
            for j,step in enumerate(_mstree_path):
                if j == 0: continue
                #Check to avoid borders printting horizontal or vertical lines
                if (_mstree_path[j-1][0] == 0 and _mstree_path[j][0] == n1-1) or (_mstree_path[j-1][0] == n1-1 and _mstree_path[j][0] == 0) or (_mstree_path[j-1][1] == 0 and _mstree_path[j][1] == n2-1) or (_mstree_path[j-1][1] == n2-1 and _mstree_path[j][1] == 0): continue
                aux = np.stack((_mstree_path[j-1],_mstree_path[j])).T
                plt.plot(aux[1], aux[0],c='w',linewidth=1)

    if allinp:
        for bmu in auxbmus:
            msptree.highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=0.5)

    plt.savefig(outname+'.pdf')
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('--allinp',help = 'highlight all input data as white squares',default = False, action='store_true')
    parser.add_argument('--unfold',help='Unfold the UMAT using the queries minsptree',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    parser.add_argument('--save',help = 'Sufix to save the local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--load',help = 'Sufix to load a precalculated local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--remap',help = 'To remap the minsptree of the fold umat to the unfold umat withour recomputing it on the uumat',default = False, action = 'store_true')
    parser.add_argument('--clust',help = '', default = False, action = 'store_true')
    args = parser.parse_args()


    main(somfile=args.som,bmusfile=args.bmus,outname=args.out,allinp=args.allinp,unfold=args.unfold,minsptree=args.minsptree,save=args.save,load=args.load,remap=args.remap,clustering=args.clust)

