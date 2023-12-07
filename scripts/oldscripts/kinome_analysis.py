sys.path.insert(1, '/work/ifilella/T6SSref')
import fastaf
import subprocess
import numpy as np

#Total sequences and different families from database
"""
data = '/work/ifilella/alignscape/data/Human_kinome/human_kinome.fasta'
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
out = '/work/ifilella/alignscape/data/Human_kinome/human_kinome.99.fasta'
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
data = '/work/ifilella/alignscape/data/Human_kinome/human_kinome_nature_inf_upper.aln'
fa = fastaf.fastaf(data,fulltitle=True)
print('Total sequences: %d'%len(fa.homolseqs))
titles = [seq.title for seq in fa.homolseqs]
families = [ title.split("_")[0] for title in titles]
print('Families: ',set(families))
"""

#Overlapings
"""
bmus = np.genfromtxt('/work/ifilella/alignscape/Kinome/90x90_200e/kinome_bmus.txt',dtype=str)
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
