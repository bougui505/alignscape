import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

kinome_data = '/home/ifilella/Projects/quicksom_seq/results/kinome_article/QE_TE_erorrs/kinome_errors.csv'
gpcrs_data = '/home/ifilella/Projects/quicksom_seq/results/gpcrs_article/QE_TE_erorrs/gpcrs_errors.csv'
tssb_data = '/home/ifilella/Projects/quicksom_seq/results/TssB_article/QE_TE_erorrs/TssB_errors.csv'
labels = ['Kinome','GPCRs','TssB']
f_QE = plt.figure(1)
f_TE = plt.figure(2)
for label,data in zip(labels,[kinome_data,gpcrs_data,tssb_data]):
    df = pd.read_csv(data)
    epochs = df.epochs.unique()
    epochs = np.sort(epochs)
    QE_means = []
    TE_means = []
    QE_std = []
    TE_std = []
    for epoch in epochs:
        epoch_df = df[df['epochs']==epoch]
        QE_means.append(epoch_df['QE'].mean())
        QE_std.append(epoch_df['QE'].std())
        TE_means.append(epoch_df['TE'].mean())
        TE_std.append(epoch_df['TE'].std())
    plt.figure(1)
    plt.errorbar(epochs, QE_means, yerr=QE_std, fmt="o",linestyle='-',label=label,markersize=2,ecolor='black',capsize=2)
    plt.figure(2)
    plt.errorbar(epochs, TE_means, yerr=TE_std, fmt="o",linestyle='-',label=label,markersize=2,ecolor='black',capsize=2)

plt.figure(1)
plt.xlabel('SOM epochs')
plt.ylabel('Average QE (10 iterations)')
plt.legend()
plt.savefig('all_QE.png')
plt.savefig('all_QE.pdf')
plt.figure(2)
plt.xlabel('SOM epochs')
plt.ylabel('Average TE (10 iterations)')
plt.legend()
plt.savefig('all_TE.png')
plt.savefig('all_TE.pdf')

#QE of TssB alone
plt.figure()
df = pd.read_csv(tssb_data)
epochs = df.epochs.unique()
epochs = np.sort(epochs)
QE_means = []
QE_std = []
for epoch in epochs:
    epoch_df = df[df['epochs']==epoch]
    QE_means.append(epoch_df['QE'].mean())
    QE_std.append(epoch_df['QE'].std())
plt.errorbar(epochs, QE_means, yerr=QE_std, fmt="o", linestyle='-', label='TssB', markersize=2,ecolor='black', capsize=2,color='green')
plt.xlabel('SOM epochs')
plt.ylabel('Average QE (10 iterations)')
plt.legend()
plt.savefig('TssB_QE.png')
plt.savefig('TssB_QE.pdf')
