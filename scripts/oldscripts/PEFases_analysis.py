import sys
sys.path.insert(1, '/home/ifilella/Projects/alignscape')
import fastaf
import plot_umat
import alignscape
import pickle
import quicksom.som
import functools
import matplotlib.pyplot as plt
import minsptree as mspt

#Input data
somfile1 = '/home/ifilella/Projects/alignscape/results/PEFASES/PEFall/round1/som.pickle'
somfile2 = '/home/ifilella/Projects/alignscape/results/PEFASES/PEFall/round2/som.pickle'
somfilePET = '/home/ifilella/Projects/alignscape/results/PEFASES/PETonly/round2/som.pickle'
peledata = '/home/ifilella/Projects/alignscape/data/PEFASES/Final_ranking_PEF3.csv'
outliersfile = '/home/ifilella/Projects/alignscape/data/PEFASES/alignments/Final_fasta_PEF3.round1_output_rejected.fasta'
glidedata= '/home/ifilella/Projects/alignscape/data/PEFASES/Glide_score_PEF3.csv'
bedata = '/home/ifilella/Projects/alignscape/data/PEFASES/ranking_binding_energy.csv'
oxydata = '/home/ifilella/Projects/alignscape/data/PEFASES/ranking_oxyanion.csv'
mhet = '/home/ifilella/Projects/alignscape/data/PEFASES/MHET-like_sequences.fasta'
pet = '/home/ifilella/Projects/alignscape/data/PEFASES/PET-like_sequences.fasta'
fsc = '/home/ifilella/Projects/alignscape/data/PEFASES/FsC-like_sequences.fasta'


#Load PELE data (~1500)
datadic = {}
f = open(peledata,'r')
for line in f:
    line = line.replace('\n','').replace(' ','').split(',')
    datadic[line[0]]=float(line[1])
f.close()

#Load Glide data (~1500)
glidedic ={}
f = open(glidedata,'r')
for line in f:
    line = line.replace('\n','').split(',')
    glidedic[line[0]]=float(line[1])
f.close()

#Load BE data (63)
bedic = {}
f = open(bedata,'r')
for line in f:
    line = line.replace('\n','').split(',')
    bedic[line[0]]=float(line[1])
f.close()

#Load oxy data (63)
oxydic = {}
f = open(oxydata,'r')
for line in f:
    line = line.replace('\n','').split(',')
    oxydic[line[0]]=float(line[1])
f.close()

#Load outliers
outliers = []
f = open(outliersfile,'r')
for line in f:
    if '>' in line:
        line = line.replace('>','').replace('\n','')
        outliers.append(line)

#Load MHET, PET and FsC titiles
MHET = fastaf.Alignment(mhet).names
PET = fastaf.Alignment(pet).names
FSC = fastaf.Alignment(fsc).names

#Load SOM round1, round2, PET, MHET and FsC
with open(somfile1, 'rb') as somfileaux:
    som1 = pickle.load(somfileaux)
with open(somfile2, 'rb') as somfileaux:
    som2 = pickle.load(somfileaux)
with open(somfilePET, 'rb') as somfileaux:
    somPET = pickle.load(somfileaux)
b62 = alignscape.get_blosum62()
som1.metric = functools.partial(alignscape.seqmetric, b62=b62)
bmus1 = list(zip(*som1.bmus.T))
bmus2 = list(zip(*som2.bmus.T))
bmusPET = list(zip(*somPET.bmus.T))
titles1 = som1.labels
titles2 = som2.labels
titlesPET = somPET.labels
titles1 = [title.replace('>','') for title in titles1]
titles2 = [title.replace('>','') for title in titles2]
titlesPET = [title.replace('>','') for title in titlesPET]
peleZ1 = [datadic[title] for title in titles1]
peleZ2 = [datadic[title] for title in titles2]
pelePET = []
pelePETtitles = []
for title in titlesPET:
    try:
        pelePET.append(datadic[title])
        pelePETtitles.append(title)
    except:
        print(title)
#pelePET = [datadic[title] for title in titlesPET]
peleZoutliers = [datadic[outlier] for outlier in outliers]
glideZ1 = [glidedic[title] for title in titles1]
glideZ2 = [glidedic[title] for title in titles2]
glidePET = []
glidePETtitles = []
for title in titlesPET:
    try:
        glidePET.append(glidedic[title])
        glidePETtitles.append(title)
    except:
        print(title)
#glidePET = [glidedic[title] for title in titlesPET]
glideZoutliers = [glidedic[outlier] for outlier in outliers]
beZ1 = []
betitles1 = []
for title in titles1:
    try:
        beZ1.append(bedic[title])
        betitles1.append(title)
    except:
        pass
oxyZ1 = []
oxytitles1 = []
for title in titles1:
    try:
        oxyZ1.append(oxydic[title])
        oxytitles1.append(title)
    except:
        pass
mhettitles = [title for title in titles1 if title in MHET]
pettitles = [title for title in titles1 if title in PET]
fsctitles = [title for title in titles1 if title in FSC]

print(len(MHET),len(mhettitles))
print(len(PET),len(pettitles))
print(len(FSC),len(fsctitles))

#To plot SOM
rows1 = list(zip(*bmus1))[1]
rows2 = list(zip(*bmus2))[1]
rowsPET = list(zip(*bmusPET))[1]
rows_dic1 = dict(zip(titles1, rows1))
rows_dicPET = dict(zip(titlesPET, rowsPET))
rows_PETPELE = [rows_dicPET[title] for title in pelePETtitles]
rows_out = [rows_dic1[outlier] for outlier in outliers]
rows_BE = [rows_dic1[title] for title in betitles1]
rows_OXY = [rows_dic1[title] for title in oxytitles1]
rows_MHET = [rows_dic1[title] for title in mhettitles]
rows_PET = [rows_dic1[title] for title in pettitles]
rows_FSC = [rows_dic1[title] for title in fsctitles]
cols1 = list(zip(*bmus1))[0]
cols2 = list(zip(*bmus2))[0]
colsPET = list(zip(*bmusPET))[0]
cols_dic1 = dict(zip(titles1, cols1))
cols_dicPET = dict(zip(titlesPET, colsPET))
cols_PETPELE = [cols_dicPET[title] for title in pelePETtitles]
cols_out = [cols_dic1[outlier] for outlier in outliers]
cols_BE = [cols_dic1[title] for title in betitles1]
cols_OXY = [cols_dic1[title] for title in oxytitles1]
cols_MHET = [cols_dic1[title] for title in mhettitles]
cols_PET = [cols_dic1[title] for title in pettitles]
cols_FSC = [cols_dic1[title] for title in fsctitles]

"""
#Plot U-matrix alone
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
figure.colorbar(cax)
plt.savefig('PEFases_umat_r1.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som2.umat)
figure.colorbar(cax)
plt.savefig('PEFases_umat_r2.png')
"""

"""
#Distribution of PELE and glide values
plt.figure()
plt.hist(peleZ1,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_PELEhist_r1.png')
plt.figure()
plt.hist(peleZ2,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_PELEhist_r2.png')
plt.figure()
plt.hist(peleZoutliers,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_PELEhist_outliers.png')
plt.figure()
plt.hist(glideZ1,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_glidehist_r1.png')
plt.figure()
plt.hist(glideZ2,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_glidehist_r2.png')
plt.figure()
plt.hist(glideZoutliers,bins=15,alpha=0.8,rwidth=0.95)
plt.savefig('PEFases_glidehist_outliers.png')
"""

"""
#Plot UMATS with PELE data
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows1,cols1,c=peleZ1,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_PELEreumat_r1.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som2.umat)
zax = ax.scatter(rows2,cols2,c=peleZ2,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_PELEreumat_r2.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows1,cols1,c=peleZ1,cmap='binary',s=8)
ax.scatter(rows_out,cols_out,c='r',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_PELEreumat_outliers_r1.png')
"""
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(somPET.umat)
zax = ax.scatter(rows_PETPELE,cols_PETPELE,c=pelePET,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PET_PELEreumat.png')

"""
#PLOT UMAT with Glide data
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows1,cols1,c=glideZ1,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_GLIDEreumat_r1.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som2.umat)
zax = ax.scatter(rows2,cols2,c=glideZ2,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_GLIDEreumat_r2.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows1,cols1,c=glideZ1,cmap='binary',s=8)
ax.scatter(rows_out,cols_out,c='r',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_GLIDEreumat_outliers_r1.png')
"""

"""
#PLOT UMAT with BE and oxy data
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows_BE,cols_BE,c=beZ1,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_BEreumat_r1.png')

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
zax = ax.scatter(rows_OXY,cols_OXY,c=oxyZ1,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_OXYreumat_r1.png')
"""

"""
#Color dots by type of sequence (MHET, PET or FSC)
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(som1.umat)
ax.scatter(rows_MHET,cols_MHET,c='orange',s=8)
ax.scatter(rows_PET,cols_PET,c='cyan',s=8)
ax.scatter(rows_FSC,cols_FSC,c='lime',s=8)
figure.colorbar(cax)
plt.savefig('PEFases_typesreumat_r1.png')
"""


"""
#Get clusterized plots for PEFases
figure = plt.figure()
ax = figure.add_subplot(111)
umat_clst1 = mspt.get_clusterized_umat(som1.umat,som1.adj,som1.umat.shape,min_distance=15)
cax = ax.matshow(umat_clst1)
zax = ax.scatter(rows1,cols1,c=peleZ1,cmap='binary',s=8)
ax.scatter(rows_out,cols_out,c='r',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_reumat_outliers_clst_r1.png')
f = open('PEFases_reumat_outliers_clst_r1.txt','w')
f.write('#bmu_r #bmu_c #title #cluster\n')
for i,bmu in enumerate(bmus1):
    bmu_r = bmus1[i][0]
    bmu_c = bmus1[i][1]
    f.write('%d %d %s %d\n'%(bmu_r,bmu_c,titles1[i], umat_clst1[bmu_r][bmu_c]))
f.close()

figure = plt.figure()
ax = figure.add_subplot(111)
umat_clst2 = mspt.get_clusterized_umat(som2.umat,som2.adj,som2.umat.shape,min_distance=3)
cax = ax.matshow(umat_clst2,cmap='tab20b')
zax = ax.scatter(rows2,cols2,c=peleZ2,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PEFases_reumat_outliers_clst_r2.png')
f = open('PEFases_reumat_outliers_clst_r2.txt','w')
f.write('#bmu_r #bmu_c #title #cluster\n')
for i,bmu in enumerate(bmus2):
    bmu_r = bmus2[i][0]
    bmu_c = bmus2[i][1]
    f.write('%d %d %s %d\n'%(bmu_r,bmu_c,titles2[i], umat_clst2[bmu_r][bmu_c]))
f.close()
"""

#Get clusterized plots for PETases
figure = plt.figure()
ax = figure.add_subplot(111)
umat_clstPET = mspt.get_clusterized_umat(somPET.umat,somPET.adj,somPET.umat.shape,min_distance=8)
cax = ax.matshow(umat_clstPET,cmap='tab20b')
zax = ax.scatter(rows_PETPELE,cols_PETPELE,c=pelePET,cmap='binary',s=8)
figure.colorbar(cax)
figure.colorbar(zax)
plt.savefig('PET_PELEreumat_clst.png')
