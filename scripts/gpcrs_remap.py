import numpy as np
import sys
import matplotlib.pyplot as plt
from adjustText import adjust_text
import pickle
import quicksom.som
import quicksom.somax
import functools
sys.path.insert(2, '/work/ifilella/quicksom_seq')
import som_seq
import jax_imports
import seqdataloader as seqdataloader
import itertools
import scipy
import scipy.sparse.csgraph as csgraph
import minsptree as msptree

def main(somfile,bmusfile,outname='reumat.pdf',unfold=False,minsptree=False,save_localadj=save_localadj):
    
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

    #Get the regular or the unfold umat
    if unfold:
        som.compute_umat(unfold=True,normalize=True)
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
                plt.plot(aux[1], aux[0],c='w',linewidth=1)

    texts = []
    setoverlapping = ('Q8WXD0', 'Q96LB2','Q16581', 'Q86VZ1','O14718', 'Q99678')
    #Colour the BMU of the initial data
    for k, bmu in enumerate(auxbmus):
        if subtypes[k] == 'A-alpha':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="darkviolet", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='darkviolet',s=5)
        if subtypes[k] == 'A-beta':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="mediumpurple", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='mediumpurple',s=5)
        if subtypes[k] == 'A-gamma':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="plum", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='plum',s=5)
        if subtypes[k] == 'A-delta':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="magenta", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='magenta',s=5)
        if subtypes[k] == 'A-other':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="lavenderblush", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='lavenderblush',s=5)
        if subtypes[k] == 'Olfactory':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="mediumaquamarine", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='mediumaquamarine',s=5)
        if subtypes[k] == 'Taste2':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="lime", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='lime',s=5)
        if subtypes[k] == 'Vomeronasal':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="olive", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='olive',s=5)
        if subtypes[k] == 'B':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="yellow", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='yellow',s=5)
        if subtypes[k] == 'Adhesion':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="black", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='black',s=5)
        if subtypes[k] == 'C':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="orange", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='orange',s=5)
        if subtypes[k] == 'F':
            #highlight_cell(int(bmu[1]),int(bmu[0]), color="red", linewidth=1)
            plt.scatter(bmu[1], bmu[0],c='red',s=5)
        #if labels[k] in setoverlapping:
        #    texts.append(plt.text(int(bmu[1]), int(bmu[0]),labels[k] ,fontsize=8,c='white'))
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.savefig(outname,dpi=500)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('--unfold',help='Unfold the UMAT',default = False, action = 'store_true')
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between queries', default = False, action = 'store_true')
    parser.add_argument('-save_localadj',help = 'To save the local adj matrix',default = None, type = str)
    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,outname=args.out,unfold=args.unfold,minsptree=args.minsptree,save_localadj=args.save_localadj)
