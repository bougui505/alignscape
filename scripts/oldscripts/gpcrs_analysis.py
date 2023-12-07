sys.path.insert(1, '../../T6SSref')
import fastaf
import numpy as np

"""
#Data
data = '/work/ifilella/alignscape/data/Human_gpcr/pcbi.1004805.s002.csv'
alif = '/work/ifilella/alignscape/data/Human_gpcr/gross-alignment.aln'

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
ali.print_fasta('/work/ifilella/alignscape/data/Human_gpcr/gpcrs_human.aln')
"""

#Print all families
"""
faf = '/work/ifilella/alignscape/data/Human_gpcr/gpcrs_human_inf.99.aln'
fa = fastaf.fastaf(faf,fulltitle=True)
print('Total sequences: %d'%len(fa.homolseqs))
titles = [seq.title for seq in fa.homolseqs]
families = [ title.split("_")[-1] for title in titles]
print('Families: ',set(families))
"""

#Overlapings
"""
bmus = np.genfromtxt('/work/ifilella/alignscape/GPCRs/90x90_200e/gpcrs_bmus.txt',dtype=str)
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
