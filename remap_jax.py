import quicksom.som
import quicksom.somax
import functools
import os
import seqdataloader as seqdataloader
import numpy as np
import torch
import dill as pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text
import ast
from random import randint
import som_seq
import jax_imports
import scipy
import scipy.sparse.csgraph as csgraph
import itertools

pltcolorlist = ['r','b','g','m','orange','y','m','w']

def get_shortestPath(graph,start,end):
    sdist, pred = csgraph.shortest_path(graph, directed=False, indices = (start,end), return_predecessors=True)
    path=[]
    prev = end
    path.append(end)
    while prev != start:
        prev = pred[0][prev]
        path.append(prev)
    return path

def get_pathDist(graph,path):
    dist = 0
    for step in zip(path, path[1:]):
        dist += graph.todok()[step[0],step[1]]
    return dist

def seqmetric(seqs1, seqs2, b62):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    scores = score_matrix_vec(seqs1, seqs2, b62=b62)
    return -scores

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def _main_unfold():
    pass

def main(somfile,bmusfile,queriesfile,outname='reumat.pdf',delimiter=None,subtypes=None,allinp=False,unfold=False,minsptree=False):

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

    if unfold:
        som.compute_umat(unfold=True)
        unfbmus = [som.mapping[bmu] for bmu in bmus]
        auxbmus = unfbmus
        plt.matshow(som.uumat)
    else:
        auxbmus = bmus
        plt.matshow(som.umat)
    
    plt.colorbar()
    
    if minsptree:
        #Get all paths and path distances for all combinations of queries and generate a new graph of shortest distances between queries
        if unfold:
            n1, n2 = som.uumat.shape
            som._get_unfold_adj()
        else:
            n1, n2 = som.umat.shape
        
        indxbmus = [np.ravel_multi_index(bmu,(n1,n2)) for bmu in auxbmus]
      
       #Get a pair:index:label dicctionary
        labeldic = {}
        for i,auxbmu in enumerate(auxbmus):
            labeldic[indxbmus[i]] = (labels[i],auxbmu)
       
        #Get a local graph representing the shortest distances between queries
        localadj= {'data': [], 'row': [], 'col': []}
        paths = {}
        checkpairs = []
        for pair in itertools.permutations(indxbmus, 2):
            if pair not in checkpairs and (pair[1],pair[0]) not in checkpairs:
                #print(pair,labeldic[pair[0]][0],labeldic[pair[1]][0],labeldic[pair[0]][1],labeldic[pair[1]][1])
                checkpairs.append(pair)
            else:
                continue
            localadj['row'].extend([pair[0],pair[1]])
            localadj['col'].extend([pair[1],pair[0]])
            print('Computing shortest path between: %s %s'%(labeldic[pair[0]][0],labeldic[pair[1]][0]))
            if unfold:
                path = get_shortestPath(som.uadj,pair[0], pair[1])
            else:
                path = get_shortestPath(som.adj,pair[0], pair[1])
            paths[pair] = path
            paths[(pair[1],pair[0])] = path
            #print('Computing the length of the shortest path between: %s %s'%(labeldic[pair[0]][0],labeldic[pair[1]][0]))
            if unfold:
                pathDist = get_pathDist(som.uadj,path)
            else:
                pathDist = get_pathDist(som.adj,path)
            #print(pathDist)
            localadj['data'].extend([pathDist,pathDist])
        localadj = scipy.sparse.coo_matrix((localadj['data'], (localadj['row'], localadj['col'])))
        #print(localadj)
        
        #Get the minimal spaning tree of the queries
        mstree = csgraph.minimum_spanning_tree(localadj)
        #print(mstree)
        mstree_pairs = np.asarray(mstree.nonzero())
        mstree_pairs = np.vstack((mstree_pairs[0], mstree_pairs[1])).T
        for i,mstree_pair in enumerate(mstree_pairs):
            #print(mstree_pair)
            print('Printing the shortest parth between %s and %s'%(labeldic[mstree_pair[0]],labeldic[mstree_pair[1]]))
            mstree_path = paths[tuple(mstree_pair)]
            #print(mstree_path)
            _mstree_path = np.asarray(np.unravel_index(mstree_path, (n1, n2)))
            #print(_mstree_path)
            _mstree_path = np.vstack((_mstree_path[0], _mstree_path[1])).T
            #print(_mstree_path)
            for j,step in enumerate(_mstree_path):
                if j == 0: continue
                #print(step)
                #Check to avoid borders printting horizontal or vertical lines
                if (_mstree_path[j-1][0] == 0 and _mstree_path[j][0] == n1-1) or (_mstree_path[j-1][0] == n1-1 and _mstree_path[j][0] == 0) or (_mstree_path[j-1][1] == 0 and _mstree_path[j][1] == n2-1) or (_mstree_path[j-1][1] == n2-1 and _mstree_path[j][1] == 0): continue
                #print(_mstree_path[j-1],_mstree_path[j])
                aux = np.stack((_mstree_path[j-1],_mstree_path[j])).T
                #print(aux)
                plt.plot(aux[1], aux[0],c='w')

    if allinp:
        _allbmus = [(int(bmu[0]),int(bmu[1])) for bmu in allbmus]
        if unfold:
            _allunfbmus = [som.mapping[bmu] for bmu in _allbmus]
            _auxallbmus = _allunfbmus
        else:
            _auxallbmus = _allbmus
        for bmu in _auxallbmus:
            highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=1)
    
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
    parser.add_argument('--deli',help = 'Delimiter to trim the queries tittles',default = None, type = str)
    parser.add_argument('--subt',help = 'subtypes for specific coloring',default = None, type = str)
    parser.add_argument('--allinp',help = 'highlight all input data as white squares',default = False, action='store_true')
    parser.add_argument('--unfold',help='Unfold the UMAT',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,queriesfile=args.queries,outname=args.out,delimiter=args.deli,subtypes=args.subt,allinp=args.allinp,unfold=args.unfold,minsptree=args.minsptree)
    
