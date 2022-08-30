import quicksom.som
import functools
import os
import seqdataloader as seqdataloader
import numpy as np
import scipy.sparse
import torch
import dill as pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
import ast
import warnings

def seqmetric(seqs1, seqs2, b62):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    scores = score_matrix_vec(seqs1, seqs2, b62=b62)
    return -scores

def get_neighbors(i,j,X,Y,ravel=True):
    neighbors = []
    if ravel:
        iRef = np.ravel_multi_index((i % X, j % Y), (X, Y))
        jRef1 = np.ravel_multi_index(((i - 1) % X, (j - 1) % Y), (X, Y))
        jRef2 = np.ravel_multi_index(((i - 1) % X, (j) % Y), (X, Y))
        jRef3 = np.ravel_multi_index(((i - 1) % X, (j + 1) % Y), (X, Y))
        jRef4 = np.ravel_multi_index(((i) % X, (j - 1) % Y), (X, Y))
        jRef5 = np.ravel_multi_index(((i) % X, (j + 1) % Y), (X, Y))
        jRef6 = np.ravel_multi_index(((i + 1) % X, (j - 1) % Y), (X, Y))
        jRef7 = np.ravel_multi_index(((i + 1) % X, (j) % Y), (X, Y))
        jRef8 = np.ravel_multi_index(((i + 1) % X, (j + 1) % Y), (X, Y))
        neighbors = [iRef,jRef1,jRef2,jRef3,jRef4,jRef5,jRef6,jRef7,jRef8]
        return neighbors
    else:
        iRef = (i % X, j % Y)
        jRef1 = ((i - 1) % X, (j - 1) % Y)
        jRef2 = ((i - 1) % X, (j) % Y)
        jRef3 = ((i - 1) % X, (j + 1) % Y)
        jRef4 = ((i) % X, (j - 1) % Y)
        jRef5 = ((i) % X, (j + 1) % Y)
        jRef6 = ((i + 1) % X, (j - 1) % Y)
        jRef7 = ((i + 1) % X, (j) % Y)
        jRef8 = ((i + 1) % X, (j + 1) % Y)
        neighbors = [iRef,jRef1,jRef2,jRef3,jRef4,jRef5,jRef6,jRef7,jRef8]
        return neighbors


class Dmatrix(object):
    """
    Distance matrix
    """
    def __init__(self, som=None, bmus=None, queries=None, output=None, load=None, subtypes=None, delimiter=None):
        """
        """
        #Parse initial data
        self.out = output
        if subtypes != None:
            f = open(subtypes, 'r')
            contents = f.read()
            self.subtypes = ast.literal_eval(contents)
            f.close()
        else:
            self.subtypes = subtypes
        self.delimiter = delimiter       
 
        #Load an already calculated Dmatrix object
        if load is not None and all(x is None for x in [som,bmus,queries]):
            self.load = load
            self.df = pickle.load(open(load,'rb'))
            self.columns=self.df.columns
        #Calculate a Dmatrix using a list of queries, a som and a list of bmus
        elif all(x is not None for x in [som,bmus,queries])  and load is None:
            with open(som, 'rb') as somfile:
                self.som = pickle.load(somfile)
            allbmus = np.genfromtxt(bmus, dtype=str, skip_header=1)
            self.bmus = list()
            allqueries = open(queries,'r')
            self.queries = list()
            for query in allqueries:
                query = query.replace("\n","")
                indx = np.where(np.char.find(allbmus[:,-1,], query)>=0)
                if len(indx[0]) == 0:
                    continue
                elif len(indx[0]) > 1:
                    warnings.warn(f'The query: %s is find multiple times in the original ali'%query)
                indx = indx[0][0]
                self.bmus.append((int(allbmus[indx][0]),int(allbmus[indx][1])))
                self.queries.append(query.replace(">",""))
            self.umat = self.som.umat 
            self.dim = self.som.umat.shape[0]
            self.get_SOMDmatrix()
        else:
            raise ValueError("To initialize a Dmatrix either use 'load' or 'som+bmus+queries'")        


    def get_SOMDmatrix(self):
        """
        """
        #Obtain the row/column names for the Dmatrix
        self.columns = list()
        
        if self.subtypes != None:       
            for i,bmu in enumerate(self.bmus):
                aux = self.queries[i].split(self.delimiter)[0]
                self.columns.append(aux + " " + self.subtypes[aux])
        else:
            for i, bmu in enumerate(self.bmus):
                column = self.queries[i].split(self.delimiter)[0]
                self.columns.append(column)
        
        #Calculate the distance matrix between all conected cells of the umat
        row_list=[]
        col_list=[]
        data_list=[]
        for i in range(self.dim):
            row_list.append(np.ravel_multi_index((i,i),(self.dim,self.dim)))
            col_list.append(np.ravel_multi_index((i,i),(self.dim,self.dim)))
            data_list.append(0.0)
            for j in range(self.dim):
                c_uval=self.umat[i,j]
                ngs=get_neighbors(i,j,self.dim,self.dim,ravel=False)[1:]
                for ng in ngs:
                    row_list.append(np.ravel_multi_index(ng,(self.dim,self.dim)))
                    col_list.append(np.ravel_multi_index((i,j),(self.dim,self.dim)))
                    n_uval=self.umat[ng]
                    data_list.append(np.linalg.norm(c_uval-n_uval))
        adjmat=scipy.sparse.coo_matrix((data_list,(row_list,col_list)),shape=(self.dim*self.dim,self.dim*self.dim))
        c_distmat=scipy.sparse.csgraph.shortest_path(adjmat,directed=False)
        
        #Get the dataframe where the queries are the columns/rows and the values the shortest path between its corresponding umat cells
        data = np.zeros((len(self.columns),len(self.columns)))
        for i,b1 in enumerate(self.bmus):
            for j,b2 in enumerate(self.bmus):
                r_b1 = np.ravel_multi_index(b1,(self.dim,self.dim))
                r_b2 = np.ravel_multi_index(b2,(self.dim,self.dim))
                data[i,j] = c_distmat[r_b1,r_b2]
        self.df=pd.DataFrame(data,columns=self.columns,index=self.columns)

    def save(self,out=None):
        if out == None and self.out != None:
            self.df.to_pickle(self.out+'.p')
        elif out!=None:
            self.df.to_pickle(out+'.p')
        else:
            self.df.to_pickle('dmatrix.p')

    def plot(self,out=None):
        #Color rows and columns by subtype
        if len(self.columns) == 0:
            warnings.warn(f'There are no queries', UserWarning)
            return 0
        elif len(self.columns) == 1:
            cg = sns.heatmap(self.df,cmap="RdBu_r",linewidths = 0.30,yticklabels=True,xticklabels=True)
        elif len(self.columns) > 1:
            if self.subtypes != None:
                used_subtypes = list(set(self.subtypes.values()))
                used_subtypes.sort()
                subtypes_pal = sns.color_palette("Set1", n_colors=len(used_subtypes), desat=.99)
                subtypes_lut = dict(list(zip(list(map(str, used_subtypes)), subtypes_pal)))
                colors = []
                for i,name in enumerate(zip(self.columns,self.columns)):
                    subtype=name[0].split(" ")[1]
                    colors.append(subtypes_lut[subtype])
                dfcolors=pd.DataFrame({'subtype':colors},index=self.columns)
                cg = sns.clustermap(self.df,cmap="RdBu_r",linewidths = 0.30,metric='cityblock',col_colors=dfcolors, row_colors=dfcolors,yticklabels=True,xticklabels=True)
                #Add subtype legend
                for label in used_subtypes:
                    cg.ax_col_dendrogram.bar(0, 0, color=subtypes_lut[label],
                                            label=label, linewidth=0)
                    cg.ax_col_dendrogram.legend(loc="best", bbox_to_anchor=(0, 1.2) ,ncol=1)
            else:
                cg = sns.clustermap(self.df,cmap="RdBu_r",linewidths = 0.30,metric='cityblock',yticklabels=True,xticklabels=True)

        #Save the plot
        if out == None and self.out != None:
            plt.savefig("%s.pdf"%self.out,dpi = 300)
        elif out != None:
            plt.savefig("%s.pdf"%out,dpi = 300)
        else:
            plt.savefig("dmatrix.pdf",dpi = 300)

    def update(self,dmatrix2):
        """
        """
        dict1 = self.df.to_dict()
        dict2 = dmatrix2.df.to_dict()
        for ki in dict2.keys():
            for kj in dict2[ki].keys():
                if ki in dict1.keys() and kj in dict1.keys(): dict1[ki][kj]=dict2[ki][kj]
        self.df=pd.DataFrame(dict1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--som', help = 'Som file', required = True)
    parser.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the SOM', required = True)
    parser.add_argument('-q', '--queries', help = 'Sequences to be remmaped',required = True)
    parser.add_argument('-o', '--out', help = 'Output name for the dmatrix plot and pickle file',default='dmatrix')
    parser.add_argument('--subt',help = 'Subtype dicctionary',default = None)
    parser.add_argument('--deli',help = 'Delimiter to trim the queries tittles',default = None, type = str)
    args = parser.parse_args()

    dmatrix = Dmatrix(som=args.som,
            bmus=args.bmus,
            queries=args.queries,
            output=args.out,
            subtypes=args.subt,
            delimiter=args.deli)
    dmatrix.plot()
    dmatrix.save()
