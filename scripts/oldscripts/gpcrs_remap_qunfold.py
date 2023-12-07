import numpy as np
import sys
import matplotlib.pyplot as plt
from adjustText import adjust_text
import pickle
import quicksom.som
import quicksom.somax
import functools
sys.path.insert(2, '/work/ifilella/alignscape')
import som_seq
import jax_imports
import seqdataloader as seqdataloader
import itertools
import scipy
import scipy.sparse.csgraph as csgraph
import minsptree as msptree

def main(somfile,bmusfile,outname='reumat.pdf',unfold=False,minsptree=False,save=None,load=None,remap=False):

    #Data loading
    allbmus = np.genfromtxt(bmusfile, dtype=str, skip_header=1)
    with open(somfile, 'rb') as somfileaux:
            som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)

    #Parse the data
    labels = list()
    subtypes = list()
    bmus = list()
    for k,bmu in enumerate(allbmus):
        bmus.append((int(bmu[0]),int(bmu[1])))
        labels.append(bmu[-1].replace(">","").split("_")[0])
        subtypes.append(bmu[-1].split("_")[-1])

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

    #Get the regular or the unfold umat
    if unfold:

        #Get the mininimal spanning tree of the localadj matrix between the queries bmus
        mstree = csgraph.minimum_spanning_tree(localadj)

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
        #Get the minimal spaning tree of the queries
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
                for k in paths:
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

    texts = []
    setoverlapping = ('Q8WXD0', 'Q96LB2','Q16581', 'Q86VZ1','O14718', 'Q99678')
    #Colour the BMU of the initial data
    for k, bmu in enumerate(auxbmus):
        if subtypes[k] == 'A-alpha':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="darkviolet", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='darkviolet',s=7)
        if subtypes[k] == 'A-beta':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="mediumpurple", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='mediumpurple',s=7)
        if subtypes[k] == 'A-gamma':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="plum", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='plum',s=7)
        if subtypes[k] == 'A-delta':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="magenta", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='magenta',s=7)
        if subtypes[k] == 'A-other':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="lavenderblush", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='lavenderblush',s=7)
        if subtypes[k] == 'Olfactory':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="mediumaquamarine", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='mediumaquamarine',s=7)
        if subtypes[k] == 'Taste2':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="lime", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='lime',s=7)
        if subtypes[k] == 'Vomeronasal':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="olive", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='olive',s=7)
        if subtypes[k] == 'B':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="yellow", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='yellow',s=7)
        if subtypes[k] == 'Adhesion':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="black", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='black',s=7)
        if subtypes[k] == 'C':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="orange", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='orange',s=7)
        if subtypes[k] == 'F':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="red", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='red',s=7)
        #if labels[k] in setoverlapping:
        #    texts.append(plt.text(int(bmu[1]), int(bmu[0]),labels[k] ,fontsize=8,c='white'))
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.savefig(outname,dpi=500)
    #plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('--unfold',help='Unfold the UMAT',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    parser.add_argument('--save',help = 'Sufix to save the local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--load',help = 'Sufix to load a precalculated local adj matrix of the BMUs of the queries and its paths',default = None, type = str)
    parser.add_argument('--remap',help = 'To remap the minsptree of the fold umat to the unfold umat withour recomputing it on the uumat',default = False, action = 'store_true')
    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,outname=args.out,unfold=args.unfold,minsptree=args.minsptree,save=args.save,load=args.load,remap=args.remap)
