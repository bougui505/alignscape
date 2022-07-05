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

pltcolorlist = ['r','b','g','m','orange','y','m','w']

def main(somfile,bmusfile,queriesfile,outname='reumat.pdf',delimiter=None,subtypes=None,allinp=False,unfold=False,minsptree=False,save=None,load=None,remap=False):

    #Load the data (allbmus, the queries, the som and the subtype dicc)
    allbmus = np.genfromtxt(bmusfile, dtype=str, skip_header=1)
    queries = open(queriesfile,'r')
    with open(somfile, 'rb') as somfileaux:
            som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
    if subtypes != None:
        f = open(subtypes, 'r')
        contents = f.read()
        dsubtypes = ast.literal_eval(contents)
        f.close() 
    
    #Associate a color for each subtype
    csubtypes = pltcolorlist[0:len(set(dsubtypes.values()))]
    _ksubtypes = dsubtypes.values()
    ksubtypes = sorted(list(set(_ksubtypes)))
    dcolors = dict(zip(ksubtypes,csubtypes)) 

    #Parse the queries and their corresponding bmus
    labels = list()
    bmus = list()

    #Get the cells of the queries and parse their titles
    for query in queries:
        query = query.replace("\n","")
        for bmu in allbmus:
            if query in bmu[-1]:
                if delimiter != None:
                    aux = query.replace(">","").split(delimiter)[0]
                else:
                    aux = query
                labels.append(aux)
                bmus.append((int(bmu[0]),int(bmu[1])))

    #Load or compute the localadj matrix between the qbmus
    if load:
        try:
            localadj = msptree.load_localadjmat(load + '_localadj.npz')
        except:
            raise KeyError('%s_localadj.npz is missing or have a wrong name'%load)
        try:
            with open(load+'_paths.pkl', 'rb') as f:
                paths = pickle.load(f)
        except:
            raise KeyError('%s_paths.pkl is missing or have a wrong name'%load)
    else:
        localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus,verbose=True)
        if save is not None:
            scipy.sparse.save_npz(save + '_localadj.npz', localadj)
            with open(save + '_paths.pkl', 'wb') as f:
                pickle.dump(paths, f)


    if unfold:
        
        #Get the mininimal spanning tree of the localadj matrix between the queries bmus
        mstree = graph.minimum_spanning_tree(localadj)
        
        #Use the minimial spanning three between queries bmus to unfold the umat
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
    plt.matshow(auxumat)
    plt.colorbar()
    
    if minsptree:
        #Get the minimal spanning tree of the queries
        if unfold:
            if remap:
                mstree_pairs, paths = msptree.get_minsptree(localadj,paths)
                _n1,_n2 = som.umat.shape
                unf_mstree_pairs = []
                unf_rpaths = {}
                for pair in mstree_pairs:
                    unf_rpair = [msptree.get_uumat_ravel_cell(pair[0],(_n1,_n2),(n1,n2),mapping), msptree.get_uumat_ravel_cell(pair[1],(_n1,_n2),(n1,n2),mapping)]
                    unf_mstree_pairs.append(unf_rpair)
                unf_mstree_pairs = np.asarray(unf_mstree_pairs)
                mstree_pairs = unf_mstree_pairs
                for k in paths:
                    unf_rk = (msptree.get_uumat_ravel_cell(k[0],(_n1,_n2),(n1,n2),mapping),msptree.get_uumat_ravel_cell(k[1],(_n1,_n2),(n1,n2),mapping))
                    unf_rpath = [msptree.get_uumat_ravel_cell(step,(_n1,_n2),(n1,n2),mapping) for step in paths[k]]
                    unf_rpaths[unf_rk] = unf_rpath
                paths = unf_rpaths
            else:
                ulocaladj, upaths = msptree.get_localadjmat(auxumat,auxadj,auxbmus,verbose=True)
                mstree_pairs, paths = msptree.get_minsptree(ulocaladj,upaths)
        else:
             mstree_pairs, paths = msptree.get_minsptree(localadj,paths)
       
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
        _allbmus = [(int(bmu[0]),int(bmu[1])) for bmu in allbmus]
        if unfold:
            _allunfbmus = [mapping[bmu] for bmu in _allbmus]
            _auxallbmus = _allunfbmus
        else:
            _auxallbmus = _allbmus
        for bmu in _auxallbmus:
            msptree.highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=0.5)
    
    texts=[]
    for i, bmu in enumerate(auxbmus):
        print(bmu,labels[i])
        if bmu[1]==0 and bmu[0]!=0:
            plt.scatter(bmu[1]+1, bmu[0],c=dcolors[dsubtypes[labels[i]]],s=7)
        elif bmu[1]!=0 and bmu[0]==0:
            plt.scatter(bmu[1], bmu[0]+1,c=dcolors[dsubtypes[labels[i]]],s=7)
        elif bmu[1]==0 and bmu[0]==0:
            plt.scatter(bmu[1]+1, bmu[0]+1,c=dcolors[dsubtypes[labels[i]]],s=7)
        else:
            plt.scatter(bmu[1], bmu[0],c=dcolors[dsubtypes[labels[i]]],s=7)
        texts.append(plt.text(bmu[1], bmu[0], labels[i],fontsize=6,c='gainsboro'))
    adjust_text(texts,only_move={'points':'y', 'texts':'y'},arrowprops=dict(arrowstyle="->, head_width=0.2", color='gainsboro', lw=0.5))

    plt.savefig(outname+'.pdf')
    plt.show() 

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-q', '--queries', help = 'Sequences to be remmaped',required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('--deli',help = 'Delimiter to trim the queries tittles',default = None, type = str)
    parser.add_argument('--subt',help = 'subtypes for specific coloring',default = None, type = str)
    parser.add_argument('--allinp',help = 'highlight all input data as white squares',default = False, action='store_true')
    parser.add_argument('--unfold',help='Unfold the UMAT using the queries minsptree',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    parser.add_argument('--save',help = 'Sufix to save the local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--load',help = 'Sufix to load a precalculated local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--remap',help = 'To remap the minsptree of the fold umat to the unfold umat withour recomputing it on the uumat',default = False, action = 'store_true')
    args = parser.parse_args()
    

    main(somfile=args.som,bmusfile=args.bmus,queriesfile=args.queries,outname=args.out,delimiter=args.deli,subtypes=args.subt,allinp=args.allinp,unfold=args.unfold,minsptree=args.minsptree,save=args.save,load=args.load,remap=args.remap)
    
