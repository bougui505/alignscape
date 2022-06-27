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

pltcolorlist = ['r','b','g','m','orange','y','m','w']

def main(somfile,bmusfile,queriesfile,outname='reumat.pdf',delimiter=None,subtypes=None,allinp=False,unfold=False,minsptree=False,save_localadj=None):

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

    #Define variables according if unfold or fold
    if unfold:
        som.compute_umat(unfold=True,normalize=False)
        auxumat = som.uumat
        unfbmus = [som.mapping[bmu] for bmu in bmus]
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
        #Get the minimal spaning tree of the queries
        mstree_pairs, paths = msptree.get_minsptree(umat=auxumat,adjmat=auxadj,bmus=auxbmus,verbose=True,save_localadj=save_localadj)
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
                plt.plot(aux[1], aux[0],c='w')

    if allinp:
        _allbmus = [(int(bmu[0]),int(bmu[1])) for bmu in allbmus]
        if unfold:
            _allunfbmus = [som.mapping[bmu] for bmu in _allbmus]
            _auxallbmus = _allunfbmus
        else:
            _auxallbmus = _allbmus
        for bmu in _auxallbmus:
            msptree.highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=1)
    
    texts=[]
    for i, bmu in enumerate(auxbmus):
        print(bmu,labels[i])
        if bmu[1]==0 and bmu[0]!=0:
            plt.scatter(bmu[1]+1, bmu[0],c=dcolors[dsubtypes[labels[i]]])
            texts.append(plt.text(bmu[1]+1, bmu[0], labels[i],fontsize=9,c='white'))
        elif bmu[1]!=0 and bmu[0]==0:
            plt.scatter(bmu[1], bmu[0]+1,c=dcolors[dsubtypes[labels[i]]])
            texts.append(plt.text(bmu[1], bmu[0]+1, labels[i],fontsize=9,c='white'))
        elif bmu[1]==0 and bmu[0]==0:
            plt.scatter(bmu[1]+1, bmu[0]+1,c=dcolors[dsubtypes[labels[i]]])
            texts.append(plt.text(bmu[1]+1, bmu[0]+1, labels[i],fontsize=9,c='white'))
        else:
            plt.scatter(bmu[1], bmu[0],c=dcolors[dsubtypes[labels[i]]])
            texts.append(plt.text(bmu[1], bmu[0], labels[i],fontsize=9,c='white'))
    adjust_text(texts,force_text=(0.02, 0.1),force_points=(0.1, 0.2))

    plt.savefig(outname+'.pdf')
    plt.show() 

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-q', '--queries', help = 'Sequences to be remmaped',required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('-deli',help = 'Delimiter to trim the queries tittles',default = None, type = str)
    parser.add_argument('-subt',help = 'subtypes for specific coloring',default = None, type = str)
    parser.add_argument('--allinp',help = 'highlight all input data as white squares',default = False, action='store_true')
    parser.add_argument('--unfold',help='Unfold the UMAT',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    parser.add_argument('-save_localadj',help = 'To save the local adj matrix',default = None, type = str)
    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,queriesfile=args.queries,outname=args.out,delimiter=args.deli,subtypes=args.subt,allinp=args.allinp,unfold=args.unfold,minsptree=args.minsptree,save_localadj=args.save_localadj)
    
