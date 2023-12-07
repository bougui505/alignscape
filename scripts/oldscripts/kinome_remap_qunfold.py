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

    #Load the data (bmus and umat)
    allbmus = np.genfromtxt(bmusfile, dtype=str, skip_header=1)
    with open(somfile, 'rb') as somfileaux:
            som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)

    #Parse the data
    labels = list()
    bmus = list()
    subtypes = list()
    for k,bmu in enumerate(allbmus):
        bmus.append((int(bmu[0]),int(bmu[1])))
        labels.append("_".join(bmu[-1].split("_")[1:]))
        subtypes.append(bmu[-1].replace(">","").split("_")[0])

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
                plt.plot(aux[1], aux[0],c='w',linewidth=0.8)

    texts = []
    set1 = ('MAP3K7', 'MAP3K9', 'MAP3K10', 'MAP3K11', 'MAP3K12', 'MAP3K13', 'MAP3K20','MAP3K210')
    set2 = ('RPS6KA1_2','RPS6KA2_2', 'RPS6KA3_2', 'RPS6KA4_2','RPS6KA5_2','RPS6KA6_2')
    set3 = ('STK32A', 'STK32B', 'STK32C','RSKR')
    set4 = ('CSNK2A1', 'CSNK2A2','CSNK2A3')
    set5 = ('PBK')
    set6 = ('AURKA','AURKB','AURKC','CAMKK1','CAMKK2','PLK1','PLK2','PLK3','PLK4')
    set7 = ('PLK5')
    setnewother = ('RPS6KC1', 'RPS6KL1','CDC7', 'UHMK1','MOS', 'MLKL')

    #Colour the BMU of the initial data
    for k,bmu in enumerate(auxbmus):
        if subtypes[k] == 'CMGC':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="black", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='black',s=7)
        if subtypes[k] == 'CAMK':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="white", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='white',s=7)
        if subtypes[k] == 'TKL':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="red", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='red',s=7)
        if subtypes[k] == 'AGC':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="orange", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='orange',s=7)
        if subtypes[k] == 'RGC':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="pink", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='pink',s=7)
        if subtypes[k] == 'OTHER':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="yellow", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='yellow',s=7)
        if subtypes[k] == 'CK1':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="lime", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='lime',s=7)
        if subtypes[k] == 'STE':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="cyan", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='cyan',s=7)
        if subtypes[k] == 'NEK':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="magenta", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='magenta',s=7)
        if subtypes[k] == 'TYR':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="blue", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='blue',s=7)
    #    aux = bmu[-1].split("_")
    #    if len(aux) > 2:
    #        print(aux)
    #        if (aux[1]+'_'+aux[2]) in set2:
    #            print(aux[1]+'_'+aux[2])
    #            texts.append(plt.text(int(bmu[1]), int(bmu[0]), aux[1] ,fontsize=8,c='white'))
    #    if aux[1] in set7:
    #        plt.scatter(int(bmu[1]), int(bmu[0]),c='white',linewidth=1)
    #        texts.append(plt.text(int(bmu[1]), int(bmu[0]), aux[1] ,fontsize=8,c='white'))
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
