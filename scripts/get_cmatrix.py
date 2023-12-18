from alignscape.analysis import dmatrix
from alignscape.analysis import cmatrix
import pickle
import os
import pandas as pd

genes = ['TssA', 'TssB', 'TssC', 'TssE', 'TssF', 'TssG',
         'TssJ', 'TssK', 'TssL', 'TssM', 'hcp', 'ClpV', 'VgrG']

for gene in genes:
    df = dmatrix.get_phyloDmatrix(
        '../data/T6SS/%s/phylogenetictree/%s_queries_90.fasta.contree'
        % (gene, gene))
    df.to_pickle('%s_df.p' % gene)
    dm = dmatrix.Dmatrix(load='%s_df.p' % gene, delimiter='_')
    new_queries = ['_'.join(query.split('_')[0:2]) for query in dm.queries]
    dm.queries = new_queries
    dm.columns = new_queries
    dm.df.columns = new_queries
    dm.df = dm.df.set_index(pd.Index(new_queries))
    dm.save(out='../data/T6SS/%s/dmatrices/%s_dm_phylo'
            % (gene, gene))
    os.system('rm %s_df.p' % gene)
    dm.plot(out='%s_dm' % gene)

dmatrices = list()
for gene in genes:
    dmatrices.append("../data/T6SS/%s/dmatrices/%s_dm_phylo.p"
                     % (gene, gene))
cm = cmatrix.Cmatrix(dmatrices=dmatrices, labels=genes,
                     outname='T6SS_phylo_cmatrix')
cm.save()
cm.plot(vmin=0, vmax=1)

for gene in genes:
    df = dmatrix.get_b62Dmatrix('../data/T6SS/%s/%s_queries.fasta' % (gene, gene))
    df.to_pickle('%s_df.p' % gene)
    dm = dmatrix.Dmatrix(load='%s_df.p' % gene, delimiter='_')
    new_queries = [query.split('|')[0:2][0] for query in dm.queries]
    dm.queries = new_queries

    dm.columns = new_queries
    dm.df.columns = new_queries
    dm.df = dm.df.set_index(pd.Index(new_queries))
    dm.save(out='../data/T6SS/%s/dmatrices/%s_dm_b62'
            % (gene, gene))
    os.system('rm %s_df.p' % gene)
    dm.plot(out='%s_dm' % gene)

dmatrices = list()
for gene in genes:
    dmatrices.append("../data/T6SS/%s/dmatrices/%s_dm_b62.p"
                     % (gene, gene))
cm = cmatrix.Cmatrix(dmatrices=dmatrices, labels=genes,
                     outname='T6SS_b62_cmatrix')
cm.save()
cm.plot(vmin=0, vmax=1)
