import numpy as np
import sys
sys.path.insert(1, '../../T6SSref')
import fastaf
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

"""
#Data
data = '/work/ifilella/quicksom_seq/data/Human_gpcr/pcbi.1004805.s002.csv'
alif = '/work/ifilella/quicksom_seq/data/Human_gpcr/gross-alignment.aln'

#Data parsing
data = np.genfromtxt(data,dtype=str,delimiter=',',skip_header=1)
names = data[:,2]
ali = fastaf.fastaf(alif)

#Modification of the titles to include the GPCR class
filtindxs = []
for i,homolseq in enumerate(ali.homolseqs):
    ispdb=False
    if homolseq.title not in names:
        filtindxs.append(i)
    else:
        indx = (np.where(names == homolseq.title))[0][0]
        if 'xray' in data[indx][0]:
            ispdb=True
        if ispdb==True:
            homolseq.title = data[indx][1]+"_PDB_"+data[indx][3]
        else:
            homolseq.title = data[indx][1]+"_Uni_"+data[indx][3]

for i in filtindxs:
    del ali.homolseqs[i]

ali.do_shuffle_homolseqs()
ali.print_fasta('/work/ifilella/quicksom_seq/data/Human_gpcr/gpcrs_human.aln')
"""

#Print all families
"""
faf = '/work/ifilella/quicksom_seq/data/Human_gpcr/gpcrs_human_inf.99.aln'
fa = fastaf.fastaf(faf,fulltitle=True)
print('Total sequences: %d'%len(fa.homolseqs))
titles = [seq.title for seq in fa.homolseqs]
families = [ title.split("_")[-1] for title in titles]
print('Families: ',set(families))
"""

#Overlapings
"""
bmus = np.genfromtxt('/work/ifilella/quicksom_seq/GPCRs/90x90_200e/gpcrs_bmus.txt',dtype=str)
bmudic = {}
for bmu in bmus:
    key = (bmu[0],bmu[1])
    value = bmu[-1]
    if key in bmudic.keys():
        if bmudic[key].split("_")[-1] != value.split("_")[-1]:
            print("Different flavour overlapping:")
            print("(" + key[1] +  ","  + key[0]+ ") " + bmudic[key]+" "+value)
    else:
        bmudic[key] = value
"""

#Locate GPCRs sequences in the UMAT by families
def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

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

#Data loading
umat = np.load('/work/ifilella/quicksom_seq/GPCRs/90x90_200e/gpcrs_umat.npy')
bmus = np.genfromtxt('/work/ifilella/quicksom_seq/GPCRs/90x90_200e/gpcrs_bmus.txt',dtype=str)
somfile = '/work/ifilella/quicksom_seq/GPCRs/90x90_200e/gpcrs.p'
with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
unfold = False
minsptree = True
b62 = som_seq.get_blosum62()
som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
output = "/work/ifilella/quicksom_seq/plots/gpcrs_minsptree_dots.pdf"

#Parse the data
labels = list()
subtypes = list()
allbmus = list()
for k,bmu in enumerate(bmus):
    allbmus.append((int(bmu[0]),int(bmu[1])))
    labels.append(bmu[-1].replace(">","").split("_")[0])
    subtypes.append(bmu[-1].split("_")[-1])

#Get the regular or the unfold umat
if unfold:
    som.compute_umat(unfold=True)
    unfbmus = [som.mapping[bmu] for bmu in allbmus]
    auxbmus = unfbmus
    plt.matshow(som.uumat)
else:
    auxbmus = allbmus
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
    permsize = len(list(itertools.permutations(indxbmus, 2)))
    count = 0
    for pair in itertools.permutations(indxbmus, 2):
        count += 1
        print(str(count) + "/" + str(permsize))
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
        localadj['data'].extend([pathDist,pathDist])
    localadj = scipy.sparse.coo_matrix((localadj['data'], (localadj['row'], localadj['col'])))

    #Get the minimal spaning tree of the queries
    mstree = csgraph.minimum_spanning_tree(localadj)
    mstree_pairs = np.asarray(mstree.nonzero())
    mstree_pairs = np.vstack((mstree_pairs[0], mstree_pairs[1])).T
    for i,mstree_pair in enumerate(mstree_pairs):
        print('Printing the shortest parth between %s and %s'%(labeldic[mstree_pair[0]],labeldic[mstree_pair[1]]))
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
plt.savefig(output,dpi=500)
plt.show()
