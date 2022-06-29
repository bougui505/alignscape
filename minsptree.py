import scipy
import scipy.sparse.csgraph as csgraph
import scipy.sparse
import matplotlib.pyplot as plt
import itertools
import numpy as np

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

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def get_localadjmat(umat,adjmat,bmus,verbose=True):
    #Get all paths and path distances for all combinations of queries and generate a new graph of shortest distances between queries

    n1,n2 = umat.shape
    indxbmus = [np.ravel_multi_index(bmu,(n1,n2)) for bmu in bmus]

    localadj= {'data': [], 'row': [], 'col': []}
    paths = {}
    checkpairs = []
    indxbmus = list(set(indxbmus))
    size = int((len(list(itertools.permutations(indxbmus, 2))))/2)
    count = 0
    for pair in itertools.permutations(indxbmus, 2):
        if pair not in checkpairs and (pair[1],pair[0]) not in checkpairs:
            checkpairs.append(pair)
        else:
            continue
        count += 1
        if verbose:
            print(str(count) + "/" + str(size))
        localadj['row'].extend([pair[0],pair[1]])
        localadj['col'].extend([pair[1],pair[0]])
        if verbose:
            print('Computing shortest path between: %d %d'%(pair[0],pair[1]))
        path = get_shortestPath(adjmat,pair[0], pair[1])
        paths[pair] = path
        paths[(pair[1],pair[0])] = path
        if verbose:
            print('Computing the lenght of the shortest path between: %d %d'%(pair[0],pair[1]))
        pathDist = get_pathDist(adjmat,path)
        localadj['data'].extend([pathDist,pathDist])
    localadj = scipy.sparse.coo_matrix((localadj['data'], (localadj['row'], localadj['col'])))
    return localadj,paths

def load_localadjmat(localadjmat):
    localadj = scipy.sparse.load_npz(localadjmat)
    return localadj

def get_minsptree(umat,adjmat,bmus,verbose=True,save_localadj=None):
    localadj, paths = get_localadjmat(umat,adjmat,bmus,verbose=True)
    if save_localadj is not None:
        scipy.sparse.save_npz(save_localadj + '.npz', localadj)
    mstree = csgraph.minimum_spanning_tree(localadj)
    mstree_pairs = np.asarray(mstree.nonzero())
    mstree_pairs = np.vstack((mstree_pairs[0], mstree_pairs[1])).T
    return mstree_pairs,paths
