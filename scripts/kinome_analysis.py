import numpy as np
import sys
sys.path.insert(1, '/work/ifilella/T6SSref')
import fastaf
import subprocess
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

#Total sequences and different families from database
"""
data = '/work/ifilella/quicksom_seq/data/Human_kinome/human_kinome.fasta'
fa = fastaf.fastaf(data,fulltitle=True)
print('Total sequences: %d'%len(fa.homolseqs))
titles = [seq.title for seq in fa.homolseqs]
seqs = [seq.seq for seq in fa.homolseqs]
families = [ title.split(" ")[-1] for title in titles]
families2 = [family.split("/")[0].replace("(","") for family in families]
print('Families: ',set(families2))
"""

#Clustering
"""
out = '/work/ifilella/quicksom_seq/data/Human_kinome/human_kinome.99.fasta'
subprocess.run("cd-hit -i %s -o %s -c 0.99"%(data,out),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
output = subprocess.run("grep '>' %s | wc"%out, shell=True,capture_output=True)
totalseqs = int(output.stdout.split()[0])
print('Total sequences after clustering (0.99): %d'%totalseqs)
"""

#Average pairwise %ID of kinome
"""
identities = []
for i,seq1 in enumerate(seqs):
   for j,seq2 in enumerate(seqs):
       print(i,j) 
       if i<j:
            identity = fastaf.get_pairAlignment(seq1,seq2,gap_e=-0.5,gap_o=-5,alimode='muscle')[0]
            identities.append(identity)

identities = np.asarray(identities)
print(np.mean(identities))
"""

#Average pairwise %ID of T6SS
"""
data = '/work/ifilella/T6SSref/homologs/TssB/TssB.90.short.fa'
fa = fastaf.fastaf(data,fulltitle=True)
seqs = [seq.seq for seq in fa.homolseqs]
identities = []
for i,seq1 in enumerate(seqs):
   for j,seq2 in enumerate(seqs):
       print(i,j)
       if i<j:
            identity = fastaf.get_pairAlignment(seq1,seq2,gap_e=-0.5,gap_o=-5,alimode='muscle')[0]
            identities.append(identity)

identities = np.asarray(identities)
print(np.mean(identities))
"""

#Total sequences and different families from MSA from the nature paper
"""
data = '/work/ifilella/quicksom_seq/data/Human_kinome/human_kinome_nature_inf_upper.aln'
fa = fastaf.fastaf(data,fulltitle=True)
print('Total sequences: %d'%len(fa.homolseqs))
titles = [seq.title for seq in fa.homolseqs]
families = [ title.split("_")[0] for title in titles]
print('Families: ',set(families))
"""

#Overlapings
"""
bmus = np.genfromtxt('/work/ifilella/quicksom_seq/Kinome/90x90_200e/kinome_bmus.txt',dtype=str)
bmudic = {}
for bmu in bmus:
    key = (bmu[0],bmu[1])
    value = bmu[-1]
    if key in bmudic.keys():
        if bmudic[key].split("_")[0] != value.split("_")[0]:
            print("Different flavour overlapping:")
            print("(" + key[1] +  ","  + key[0]+ ") " + bmudic[key]+" "+value)
    else:
        bmudic[key] = value
"""


#Locate kinome sequences in the UMAT by families

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

#Load the data (bmus and umat)
umat = np.load('/work/ifilella/quicksom_seq/Kinome/90x90_200e_noPLK5/kinome_umat.npy')
bmus = np.genfromtxt('/work/ifilella/quicksom_seq/Kinome/90x90_200e_noPLK5/kinome_bmus.txt',dtype=str)
somfile = '/work/ifilella/quicksom_seq/Kinome/90x90_200e_noPLK5/kinome.p'
with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
unfold = False
minsptree = True
b62 = som_seq.get_blosum62()
som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
output = "/work/ifilella/quicksom_seq/plots/kinome_minsptree_dots.pdf"

#Parse the data
labels = list()
subtypes = list()
allbmus = list()
for k,bmu in enumerate(bmus):
    allbmus.append((int(bmu[0]),int(bmu[1])))
    labels.append("_".join(bmu[-1].split("_")[1:]))
    subtypes.append(bmu[-1].replace(">","").split("_")[0])

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
plt.savefig(output,dpi=500)
plt.show()
