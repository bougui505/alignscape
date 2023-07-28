import dmatrix
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import pearsonr
import pylab

class Cmatrix(object):
    """
    Correlation matrix
    """
    def __init__(self,dmatrices,labels,outname,rand=True,replacement=True,sampling=1,tolerance=math.inf):
        """
        """
        self.dmatrices = dmatrices
        #Load the Dataframe of each Dmatrix
        self.dfs = list()
        for dmatrix in self.dmatrices:
            df = pickle.load(open(dmatrix,'rb'))
            self.dfs.append(df)
        self.labels = labels
        self.outname = outname
        self.rand = rand
        self.tolerance = tolerance
        self.sampling = sampling
        self.replacement = replacement

        if len(self.dmatrices) != len(self.labels):
            raise ValueError ("Labels must have the same lenght than dmatrices")

        self.get_Cmatrix()

    def get_Cmatrix(self):
        """
        """
        #Select a random element with or without replacement
        if self.rand == True:
            rlabel = random.choices(self.labels)[0]
            ind = self.labels.index(rlabel)
            rdf = self.get_random_dmatrix(self.dfs[ind])
            if self.replacement:
                self.dfs.append(rdf)
                self.labels.append('randomDM')
            else:
                self.dfs[ind]=rdf
                self.labels[ind]='randomDM'

        #Calculate the correlation matrix
        corrma = np.zeros((len(self.labels),len(self.labels)))
        for i,n1 in enumerate(self.labels):
            for j,n2 in enumerate(self.labels):
                if i==j:
                    corrma[i][j] = 1
                    continue
                elif i>j:
                    continue
                else:
                    print(n1,n2)
                    df1 = self.dfs[i]
                    df2 = self.dfs[j]
                    r = self.get_correlation(df1,df2,n1,n2)
                    corrma[i][j] = r
                    corrma[j][i] = r
                    print(r)
        self.cdf = pd.DataFrame(corrma,columns=self.labels,index=self.labels)


    def get_random_dmatrix(self,df):
        values = df.values.flatten()
        shape = df.shape
        minv = values.min()
        maxv = values.max()
        for i,value in enumerate(values):
            if value == float(0):
                continue
            else:
                values[i] = minv + (random.random() * (maxv - minv))
        values = np.asarray(values)
        values = values.reshape((shape))
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i>=j: continue
                else: values[j][i] = values[i][j]
        rdf = pd.DataFrame(values,columns=df.columns,index=df.index)
        return rdf

    def get_correlation(self,df1,df2,n1,n2):
        """
        """
        #Check total number of common elements
        dic1=df1.to_dict()
        dic2=df2.to_dict()
        columns1=set(df1.columns)
        columns2=set(df2.columns)
        newcolumns = list(columns1.intersection(columns2))
        if len(newcolumns) < 3:
            warnings.warn(f'Not enough common elements between %s and %s (at least 3 needed)'%(n1,n2), UserWarning)
            return 0

        #Get common DFs
        ndf1,ndf2 = self.get_intersectedDFs(df1,df2)

        #Get loweer diagonal (without diagonal values)
        vec1 = ndf1.mask(np.triu(np.ones(ndf1.shape)).astype(bool)).stack()
        vec2 = ndf2.mask(np.triu(np.ones(ndf2.shape)).astype(bool)).stack()
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)

        #Sampling
        if self.sampling != 1:
            indexsampl = random.sample(range(len(vec1)),int(len(vec1)*self.sampling))
            indextoremove = list(set(range(0,len(vec1)))-set(indexsampl))
            if len(indextoremove) != 0:
                vec1 = np.delete(vec1,indextoremove)
                vec2 = np.delete(vec2,indextoremove)

        #Tolerance
        if self.tolerance != math.inf:
            difvec = abs(vec1-vec2)
            indextolerance = []
            if len(np.argwhere(difvec > self.tolerance)) != 0:
                indextolerance = np.concatenate(np.argwhere(difvec > self.tolerance))
                vec1 = np.delete(vec1,indextolerance)
                vec2 = np.delete(vec2,indextolerance)

        nvec1 = vec1
        nvec2 = vec2

        r = pearsonr(nvec1, nvec2)[0]
        if math.isnan(r):
            return 0
        return r

    def save(self):
        self.cdf.to_pickle(self.outname+'.p')

    def plot(self,vmin=None,vmax=None):
        cg = sns.clustermap(self.cdf,cmap="BrBG",linewidths = 0.30,metric='euclidean',vmin=vmin,vmax=vmax)
        pylab.savefig(self.outname+".pdf")

    def get_intersectedDFs(self,df1,df2):
        """
        """
        dic1=df1.to_dict()
        dic2=df2.to_dict()
        columns1=set(df1.columns)
        columns2=set(df2.columns)
        newcolumns = list(columns1.intersection(columns2))
        ndic1 = np.zeros((len(newcolumns),len(newcolumns)))
        ndic2 = np.zeros((len(newcolumns),len(newcolumns)))
        for i,k1 in enumerate(newcolumns):
            for j,k2 in enumerate(newcolumns):
                ndic1[i][j]=dic1[k1][k2]
                ndic2[i][j]=dic2[k1][k2]
        ndf1 = pd.DataFrame(ndic1,columns=newcolumns,index=newcolumns)
        ndf2 = pd.DataFrame(ndic2,columns=newcolumns,index=newcolumns)
        return ndf1,ndf2

if __name__ == '__main__':

    labels = ['TssA','TssB','TssC','TssE','TssF','TssG','TssK','TssL','TssM','ClpV','VgrG','hcp','TssJ']
    dmatrices = list()
    for l in labels:
        dmatrices.append("results/dmatrix/%s/%s.phylo.p"%(l,l))
    print(dmatrices)
    cmatrix = Cmatrix(dmatrices=dmatrices,labels=labels,outname='T6SS_phylo_cmatrix')
    cmatrix.save()
    cmatrix.plot(vmin=0, vmax=1)
