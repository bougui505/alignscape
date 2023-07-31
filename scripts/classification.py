import sys
sys.path.insert(1,'../')
from models import KNeighborsBMU,KNeighborsB62
import som_seq
#import quicksom.som
#import quicksom
#import pickle
import dill as pickle
import functools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plot_umat
import minsptree as mspt
from collections import OrderedDict
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import fastaf
from quicksom_seq.utils import models

if __name__ == '__main__':

    datadir = '/home/cactus/ifilella/Projects/quicksom_seq/results/'

    #Load the alignments (as a dict)
    ali_kinome = '%skinome_article/alignments_kinome/human_kinome_nature_inf_upper_noPLK5.aln'%datadir
    fasta_kinome = fastaf.Alignment(aliname=ali_kinome)
    fastadic_kinome = {title:seq for title,seq in zip(fasta_kinome.names,fasta_kinome.seqs)}
    ali_gpcrs = '%sgpcrs_article/alignments_gpcrs/gpcrs_human_inf.aln'%datadir
    fasta_gpcrs = fastaf.Alignment(aliname=ali_gpcrs)
    fastadic_gpcrs = {title:seq for title,seq in zip(fasta_gpcrs.names,fasta_gpcrs.seqs)}

    #Coloring scheme
    kinome_colors = {'CMGC':'black','CAMK':'white','TKL':'red','AGC':'orange','RGC':'pink','OTHER':'yellow','CK1':'lime','STE':'cyan','NEK':'magenta','TYR':'blue'}
    gpcrs_colors = {'A-alpha':'darkviolet','A-beta':'mediumpurple','A-gamma':'plum','A-delta':'magenta','A-other':'lavenderblush','Olfactory':'mediumaquamarine','Taste2':'lime','Vomeronasal':'olive','B':'yellow','Adhesion':'black','C':'orange','F':'red'}

    #Get and load the SOMs
    filename_kinome = '%skinome_article/kinome.p'%datadir
    som_kinome = models.load_som(filename_kinome)
    n1_kinome,n2_kinome = som_kinome.umat.shape
    filename_gpcrs = '%sgpcrs_article/gpcrs.p'%datadir
    som_gpcrs = models.load_som(filename_gpcrs)
    n1_gpcrs,n2_gpcrs = som_gpcrs.umat.shape
    exit()

    #Parse all data (titles,types and BMUs as lists and as dicts)
    titles_kinome = np.asarray([label.replace('>','') for label in som_kinome.labels])
    titles_gpcrs = np.asarray([label.replace('>','') for label in som_gpcrs.labels])
    types_kinome = np.asarray([label.replace('>','').split('_')[0] for label in som_kinome.labels])
    types_gpcrs = np.asarray([label.replace('>','').split('_')[-1] for label in som_gpcrs.labels])
    bmus_kinome = np.asarray([np.ravel_multi_index(bmu,(n1_kinome,n2_kinome)) for bmu in som_kinome.bmus])
    bmus_gpcrs = np.asarray([np.ravel_multi_index(bmu,(n1_gpcrs,n2_gpcrs)) for bmu in som_gpcrs.bmus])
    bmu_type_kinome = models.get_bmu_type_dic(bmus_kinome,types_kinome,'OTHER')
    uniq_bmus_kinome = np.asarray(list(bmu_type_kinome.keys()))
    uniq_types_kinome = np.asarray(list(bmu_type_kinome.values()))
    bmu_type_gpcrs = models.get_bmu_type_dic(bmus_gpcrs,types_gpcrs,'A-other')
    uniq_bmus_gpcrs = np.asarray(list(bmu_type_gpcrs.keys()))
    uniq_types_gpcrs = np.asarray(list(bmu_type_gpcrs.values()))
    print(f'Total sequences kinome: {len(titles_kinome)}')
    print(f'Total mapped units kinome: {len(bmu_type_kinome)}')
    print(f'Total sequences gpcrs: {len(titles_gpcrs)}')
    print(f'Total mapped units kinome: {len(bmu_type_gpcrs)}')

    #Load the distance matrices between SOM units
    dmatrix_kinome = models.load_dmatrix(som_kinome)
    dmatrix_gpcrs = models.load_dmatrix(som_gpcrs)

    #Create or load the b62 distances matrices of the original alignment
    with open('%skinome_article/kinome_matrixb62.p'%datadir,'rb') as inp:
        dmatrixb62_kinome = pickle.load(inp)
    with open('%sgpcrs_article/gpcrs_matrixb62.p'%datadir,'rb') as inp:
        dmatrixb62_gpcrs = pickle.load(inp)

    #Test to decide k of k-nearest neighbours for kinome and gpcrs at unit level
    idxs_unclass_k,idxs_class_k,types_unclass_k,types_class_k,bmus_unclass_k,bmus_class_k = models.split_data(uniq_types_kinome,uniq_bmus_kinome,'OTHER')

    idxs_unclass_g,idxs_class_g,types_unclass_g,types_class_g,bmus_unclass_g,bmus_class_g = models.split_data(uniq_types_gpcrs,uniq_bmus_gpcrs,'A-other')

    scoresBMU_k = OrderedDict()
    scoresBMU_g = OrderedDict()
    scoresB62_k = OrderedDict()
    scoresB62_g = OrderedDict()

    for i in range(40):
        print('%d ---'%(i+1))

        #Split classified data into training and testing
        bmus_train_k, bmus_test_k, types_train_k, types_test_k = train_test_split(bmus_class_k,types_class_k,test_size=0.25)
        bmus_train_g, bmus_test_g, types_train_g, types_test_g = train_test_split(bmus_class_g,types_class_g,test_size=0.25)
        print(f'%d BMUs in train and %d BMUs in test'%(len(bmus_train_k),len(bmus_test_k)))
        idx_train_k = np.hstack([np.where(bmus_kinome == bmu)[0] for bmu in bmus_train_k])
        titles_train_k = [titles_kinome[idx] for idx in idx_train_k]
        idx_test_k = np.hstack([np.where(bmus_kinome == bmu)[0] for bmu in bmus_test_k])
        titles_test_k = [titles_kinome[idx] for idx in idx_test_k]
        print(f'%d seqs in train and %d seqs in test'%(len(titles_train_k),len(titles_test_k)))
        idx_train_g = np.hstack([np.where(bmus_kinome == bmu)[0] for bmu in bmus_train_g])
        titles_train_g = [titles_kinome[idx] for idx in idx_train_g]
        idx_test_g = np.hstack([np.where(bmus_kinome == bmu)[0] for bmu in bmus_test_g])
        titles_test_g = [titles_kinome[idx] for idx in idx_test_g]

        #total_sequences_k = sum([len(np.where(bmus_kinome == bmu)[0]) for bmu in bmus_test_k])
        #total_sequences_g = sum([len(np.where(bmus_gpcrs == bmu)[0]) for bmu in bmus_test_g])

        for k in range(10):
            k+=1
            knnBMU_k = KNeighborsBMU(k)
            knnB62_k = KNeighborsB62(k)
            knnBMU_g = KNeighborsBMU(k)
            knnB62_g = KNeighborsB62(k)
            knnBMU_k.fit(dmatrix_kinome,bmus_train_k, types_train_k, bmus_unclass_k)
            knnB62_k.fit(dmatrixb62_kinome,titles_train_k)
            knnBMU_g.fit(dmatrix_gpcrs,bmus_train_g, types_train_g, bmus_unclass_g)
            knnB62_g.fit(dmatrixb62_gpcrs,titles_train_g)
            scoreBMU_k = knnBMU_k.score(bmus_test_k,types_test_k)
            scoreB62_k = knnB62_k.score(titles_test_k)
            scoreBMU_g = knnBMU_g.score(bmus_test_g,types_test_g)
            scoreB62_g = knnB62_k.score(titles_test_g)

            if k not in scoresBMU_k.keys():
                scoresBMU_k[k] = [scoreBMU_k]
            else:
                scoresBMU_k[k].append(scoreBMU_k)
            if k not in scoresBMU_g.keys():
                scoresBMU_g[k] = [scoreBMU_g]
            else:
                scoresBMU_g[k].append(scoreBMU_g)

            if k not in scoresB62_k.keys():
                scoresB62_k[k] = [scoreB62_k]
            else:
                scoresB62_k[k].append(scoreB62_k)
            if k not in scoresB62_g.keys():
                scoresB62_g[k] = [scoreB62_g]
            else:
                scoresB62_g[k].append(scoreB62_g)

            print(f'k:{k} scoreBMU_k:{scoreBMU_k} scoreBMU_g:{scoreBMU_g}')
            print(f'k:{k} scoreB62_k:{scoreB62_k} scoreB62_g:{scoreB62_g}')

    mean_scoresBMU_k = [np.mean(np.asarray(scrs)) for scrs in scoresBMU_k.values()]
    mean_scoresB62_k = [np.mean(np.asarray(scrs)) for scrs in scoresB62_k.values()]
    std_scoresBMU_k = [np.std(np.asarray(scrs)) for scrs in scoresBMU_k.values()]
    std_scoresB62_k = [np.std(np.asarray(scrs)) for scrs in scoresB62_k.values()]
    mean_scoresBMU_g = [np.mean(np.asarray(scrs)) for scrs in scoresBMU_g.values()]
    mean_scoresB62_g = [np.mean(np.asarray(scrs)) for scrs in scoresB62_g.values()]
    std_scoresBMU_g = [np.std(np.asarray(scrs)) for scrs in scoresBMU_g.values()]
    std_scoresB62_g = [np.std(np.asarray(scrs)) for scrs in scoresB62_g.values()]

    print(scoresBMU_k)
    print(scoresB62_k)
    print(scoresBMU_g)
    print(scoresB62_g)

    fig, axs = plt.subplots(2,1,figsize=(17,8))
    colors = [['C0','paleturquoise'],['C1','khaki']]
    scores = [[scoresBMU_k,scoresB62_k],[scoresBMU_g,scoresB62_g]]
    titles = ['Kinome','GPCRs']
    for j in range(2):
        labels1, data1 = [*zip(*scores[j][0].items())]
        box1 = axs[j].boxplot(data1,positions = labels1,patch_artist=True,widths=0.25,showfliers=True)
        labels2, data2 = [*zip(*scores[j][1].items())]
        labels2 = tuple([l+0.5 for l in list(labels2)])
        box2 = axs[j].boxplot(data2,positions = labels2,patch_artist=True,widths=0.25,showfliers=True)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=colors[j][0])
            plt.setp(box2[item], color=colors[j][1])
        plt.setp(box1["boxes"], facecolor="None")
        plt.setp(box1["fliers"], markeredgecolor=colors[j][0])
        plt.setp(box2["boxes"], facecolor='None')
        plt.setp(box2["fliers"], markeredgecolor=colors[j][1])
        axs[j].legend([box1["boxes"][0],box2["boxes"][0]], ['SOMseq','B62'])
        pos_labels = tuple([l+0.25 for l in list(labels1)])
        name_labels = ['k=%d'%i for i in range(1,len(labels1)+1)]
        plt.sca(axs[j])
        plt.xticks(pos_labels, name_labels)
        axs[j].set_title('%s: KNeigbours_SOMseq / KNeighbours_B62 accuracy test'%titles[j])
        axs[j].set_ylabel('accuracy')

    plt.savefig('boxplot_test.pdf')

    print('mean_scoresBMU_k')
    print(mean_scoresBMU_k)
    print('mean_scoresB62_k')
    print(mean_scoresB62_k)
    print('std_scoresBMU_k')
    print(std_scoresBMU_k)
    print('std_scoresB62_k')
    print(std_scoresB62_k)
    print('mean_scoresBMU_g')
    print(mean_scoresBMU_g)
    print('mean_scoresB62_g')
    print(mean_scoresB62_g)
    print('std_scoresBMU_g')
    print(std_scoresBMU_g)
    print('std_scoresB62_g')
    print(std_scoresB62_g)

    plt.figure()
    ks= [k for k in scoresBMU_k.keys()]
    plt.errorbar(ks,mean_scoresBMU_k,yerr=std_scoresBMU_k,fmt='o',linestyle='-', label='kinome SOM', markersize=2, ecolor='black', capsize=2,color='royalblue')
    ks= [k for k in scoresBMU_g.keys()]
    plt.errorbar(ks,mean_scoresBMU_g,yerr=std_scoresBMU_g,fmt='o',linestyle='-', label='gpcrs SOM', markersize=2, ecolor='black',capsize=2,color='orange')
    plt.ylabel("Accuracy")
    plt.xlabel("neighbors")
    plt.ylim([0.6,1.3])
    plt.legend()
    plt.title('KNeighboursBMU accuracy test (SOMseq)')
    plt.savefig('classification_test_SOM.pdf')

    plt.figure()
    ks= [k for k in scoresB62_k.keys()]
    plt.errorbar(ks,mean_scoresB62_k,yerr=std_scoresB62_k,fmt='o',linestyle='-', label='kinome B62',markersize=2, ecolor='black', capsize=2,color='paleturquoise')
    ks= [k for k in scoresB62_g.keys()]
    plt.errorbar(ks,mean_scoresB62_g,yerr=std_scoresB62_g,fmt='o',linestyle='-', label='gpcrs B62',markersize=2, ecolor='black',capsize=2,color='khaki')
    plt.ylabel("Accuracy")
    plt.xlabel("neighbors")
    plt.ylim([0.6,1.3])
    plt.legend()
    plt.title('KNeighboursB62 accuracy test (B62)')
    plt.savefig('classification_test_B62.pdf')

    plt.figure()
    ks= [k for k in scoresBMU_k.keys()]
    plt.errorbar(ks,mean_scoresBMU_k,yerr=std_scoresBMU_k,fmt='o',linestyle='-',label='kinome SOM', markersize=2, ecolor='black', capsize=2,color='royalblue')
    ks= [k for k in scoresB62_k.keys()]
    plt.errorbar(ks,mean_scoresB62_k,yerr=std_scoresB62_k,fmt='o',linestyle='-',label='kinome B62',markersize=2, ecolor='black', capsize=2,color='paleturquoise')
    plt.ylabel("Accuracy")
    plt.xlabel("neighbors")
    plt.ylim([0.6,1.3])
    plt.legend()
    plt.title('KNeighboursB62 and KNeighboursBMU accuracy test')
    plt.savefig('classification_test_kinome.pdf')

    plt.figure()
    ks= [k for k in scoresBMU_g.keys()]
    plt.errorbar(ks,mean_scoresBMU_g,yerr=std_scoresBMU_g,fmt='o',linestyle='-',label='gpcrs SOM', markersize=2, ecolor='black', capsize=2,color='orange')
    ks= [k for k in scoresB62_g.keys()]
    plt.errorbar(ks,mean_scoresB62_g,yerr=std_scoresB62_g,fmt='o',linestyle='-',label='gpcrs B62',markersize=2, ecolor='black', capsize=2,color='khaki')
    plt.ylabel("Accuracy")
    plt.xlabel("neighbors")
    plt.ylim([0.6,1.3])
    plt.legend()
    plt.title('KNeighboursB62 and KNeighboursBMU accuracy test')
    plt.savefig('classification_test_gpcrs.pdf')


    auxbmus_kinome = list(zip(*som_kinome.bmus.T))
    auxbmus_gpcrs = list(zip(*som_gpcrs.bmus.T))

    #Regular umat with MST and unclassified as Other
    plot_umat._plot_umat(som_kinome.umat,auxbmus_kinome,types_kinome, hideSeqs=False, legend=False, dic_colors = kinome_colors, dotsize=15)
    plot_umat._plot_msptree(som_kinome.msptree_pairs, som_kinome.msptree_paths, som_kinome.umat.shape)
    plt.savefig('kinome_notpredicted_umat.pdf')
    plot_umat._plot_umat(som_gpcrs.umat,auxbmus,types_gpcrs, hideSeqs=False, legend=False, dic_colors = gpcrs_colors, dotsize=15)
    plot_umat._plot_msptree(som_gpcrs.msptree_pairs, som_gpcrs.msptree_paths, som_gpcrs.umat.shape)
    plt.savefig('gpcrs_notpredicted_umat.pdf')

    #Unfold umat with MST and unclassified as Other
    uumat_kinome,mapping_kinome,reversed_mapping_kinome = mspt.get_unfold_umat(som_kinome.umat, som_kinome.adj, auxbmus_kinome, som_kinome.msptree)
    unfbmus_kinome = [mapping_kinome[bmu] for bmu in auxbmus_kinome]
    unf_msptree_pairs_kinome, unf_msptree_paths_kinome= mspt.get_unfold_msptree(som_kinome.msptree_pairs, som_kinome.msptree_paths, som_kinome.umat.shape, uumat_kinome.shape, mapping_kinome)
    plot_umat._plot_umat(uumat_kinome, unfbmus_kinome, types_kinome, hideSeqs=False, legend=False, dic_colors = kinome_colors, dotsize=15)
    plot_umat._plot_msptree(unf_msptree_pairs_kinome, unf_msptree_paths_kinome, uumat_kinome.shape)
    plt.savefig('kinome_unf_notpredicted_umat.pdf')
    uumat_gpcrs,mapping_gpcrs,reversed_mapping_gpcrs = mspt.get_unfold_umat(som_gpcrs.umat, som_gpcrs.adj, auxbmus_gpcrs, som_gpcrs.msptree)
    unfbmus_gpcrs = [mapping_gpcrs[bmu] for bmu in auxbmus_gpcrs]
    unf_msptree_pairs_gpcrs, unf_msptree_paths_gpcrs= mspt.get_unfold_msptree(som_gpcrs.msptree_pairs, som_gpcrs.msptree_paths, som_gpcrs.umat.shape, uumat_gpcrs.shape, mapping_gpcrs)
    plot_umat._plot_umat(uumat_gpcrs, unfbmus_gpcrs, types_gpcrs, hideSeqs=False, legend=False, dic_colors = gpcrs_colors, dotsize=15)
    plot_umat._plot_msptree(unf_msptree_pairs_gpcrs, unf_msptree_paths_gpcrs, uumat_gpcrs.shape)
    plt.savefig('gpcrs_unf_notpredicted_umat.pdf')

    #Predict unclassified type
    k=2
    knn = KNeighborsBMU(k)
    knn.fit(dmatrix, bmus_classified, types_classified, bmus_unclassified)
    f = open('classification.csv','w')
    for idx,bmu,title in zip(idxs_unclassified,bmus_unclassified,titles_unclassified):
        predicted_type = knn.predict(bmu)
        print(bmu,np.unravel_index(bmu,(n1,n2)),title,predicted_type)
        types[idx] = predicted_type
        #name = title.split('_')[1]
        name = title.split('_')[0]
        f.write(f'{name},{predicted_type}\n')
    f.close()

    #Plot umat with MST and unclassified with predicted type
    plot_umat._plot_umat(som.umat,auxbmus,types, hideSeqs=False,legend=False, dic_colors = kinome_colors, dotsize=15)
    plot_umat._plot_umat(som.umat,auxbmus,types, hideSeqs=False,legend=False,  dic_colors = gpcrs_colors, dotsize=15)
    plot_umat._plot_msptree(som.msptree_pairs, som.msptree_paths, som.umat.shape)
    plt.savefig('predicted_umat%d.pdf'%k)

    #Plot Unfold umat with MST and unclassified with predicted type
    plot_umat._plot_umat(uumat, unfbmus, types, hideSeqs=False, legend=False, dic_colors = kinome_colors, dotsize=15)
    plot_umat._plot_umat(uumat, unfbmus, types, hideSeqs=False, legend=False, dic_colors = gpcrs_colors, dotsize=15)
    plot_umat._plot_msptree(unf_msptree_pairs, unf_msptree_paths, uumat.shape)
    plt.savefig('unf_predicted_umat%d.pdf'%k)
