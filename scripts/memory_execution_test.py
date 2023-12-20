import random
import os
import torch
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from alignscape.utils import seqdataloader
from alignscape.align_scape import get_blosum62, torchify, seqmetric
from alignscape.utils import memory
from alignscape.quicksom import som

aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ-')

def generate_random_sequence(length, title='rand'):
    seq = ''.join(random.choices(aalist, k=length))
    title = '>%s' % title
    return title, seq


def generate_random_msa(length, nseqs, out=None):
    titles = []
    seqs = []
    for i in range(nseqs):
        title = 'rand%d' % (i+1)
        title, seq = generate_random_sequence(length, title)
        titles.append(title)
        seqs.append(seq)
    if out is not None:
        f = open('%s.fasta' % out, 'w')
        for title, seq in zip(titles, seqs):
            f.write('%s\n' % title)
            f.write('%s\n' % seq)
        f.close()
    return titles, seqs


def get_som_stats(ali, batch_size, somside, n_epoch=1):
    torch.cuda.empty_cache()
    dataset = seqdataloader.SeqDataset(ali)
    num_workers = os.cpu_count()
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             worker_init_fn=functools.partial(seqdataloader.workinit,fastafilename=ali))
    n_inp = dataset.__len__()
    dim = dataset.__dim__()
    b62 = get_blosum62()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device = memory.select_device()
    b62 = torchify(b62, device=device)
    somobj = som.SOM(somside,
                     somside,
                     n_epoch=n_epoch,
                     dim=dim,
                     alpha=0.5,
                     sigma=np.sqrt(somside * somside) / 4.0,
                     periodic=True,
                     metric=functools.partial(seqmetric, b62=b62),
                     sched='exp',
                     seed=0)
    somobj.to_device(device)
    print('n_input:', n_inp)
    print('batch_size:', batch_size)
    start_time = time.time()
    somobj.fit(dataset=dataloader,
               batch_size=batch_size,
               do_compute_all_dists=False,
               unfold=False,
               normalize_umat=False,
               sigma=np.sqrt(somside * somside) / 4.0,
               alpha=0.5,
               logfile='som.log')
    end_time = time.time()
    execution_time = end_time - start_time
    free, total = torch.cuda.mem_get_info(device)
    mem_usage = (total - free) / 1024 / 1024
    return mem_usage, execution_time


#titles, seqs = generate_random_msa(length=2500, nseqs=1000, out='ali')
#mem_usage, execution_time = get_som_stats(ali='ali.fasta',
#                                          batch_size=10,
#                                          somside=50)
#print(mem_usage)
#print(execution_time)

# GPU memory allocation test
lengths = list(range(10, 101, 10)) +\
    list(range(125, 501, 25)) +\
    list(range(500, 1501, 50)) +\
    list(range(1500, 2501, 100))

ff = open('memory_usage.csv', 'w')
ff.write('BatchSize,MSALength,SOMside,MemoryUsage\n')
for size in [1, 2, 5, 10]:
    for length in lengths:
        titles, seqs = generate_random_msa(length=length, nseqs=20, out='ali')
        mem_usage, ex_time  = get_som_stats(ali='ali.fasta',
                                            batch_size=size,
                                            somside=50)
        print('-----------------------------------')
        print(size, length, 50, mem_usage)
        print('-----------------------------------')
        ff.write('%d,%d,%d,%.3f\n' % (size, length, 50, mem_usage))
ff.close()

df = pd.read_csv('memory_usage.csv')
df1 = df[df['BatchSize'] == 1]
df2 = df[df['BatchSize'] == 2]
df5 = df[df['BatchSize'] == 5]
df10 = df[df['BatchSize'] == 10]
plt.figure()
plt.plot(df1['MSALength'], df1['MemoryUsage'],
         marker='.', linestyle='-', label='1')
plt.plot(df2['MSALength'], df2['MemoryUsage'],
         marker='.', linestyle='-', label='2')
plt.plot(df5['MSALength'], df5['MemoryUsage'],
         marker='.', linestyle='-', label='5')
plt.plot(df10['MSALength'], df10['MemoryUsage'],
         marker='.', linestyle='-', label='10')
plt.title('AlignScape GPU memory consumption')
plt.xlabel('MSA length')
plt.ylabel('GPU Memory (MB)')
plt.legend(title='Batch size')
plt.grid(True)
plt.savefig('memory_usage.pdf')

# GPU execution time test
lenghts = list(range(250, 2501, 250))
print(lenghts)
ff = open('epoch_time.csv', 'w')
ff.write('BatchSize,MSALength,SOMside,NumSeqs,MemoryUsage,AvTime5,StdTime5\n')
for size in [1, 2, 5, 10]:
    for length in lenghts:
        titles, seqs = generate_random_msa(length=length,
                                           nseqs=1000,
                                           out='ali')
        times = []
        for i in range(5):
            mem_usage, ex_time = get_som_stats(ali='ali.fasta',
                                               batch_size=size,
                                               somside=50)
            times.append(ex_time)
        times = np.asarray(times)
        av_ex_time = np.mean(times)
        std_ex_time = np.std(times)
        print('-----------------------------------')
        print(size, length, 50, 1000, mem_usage, av_ex_time, std_ex_time)
        print('-----------------------------------')
        ff.write('%d,%d,%d,%d,%.3f,%.4f,%.4f\n'
                 % (size, length, 50, 1000, mem_usage,
                    av_ex_time, std_ex_time))
ff.close()

df = pd.read_csv('epoch_time.csv')
df1 = df[df['BatchSize'] == 1]
df2 = df[df['BatchSize'] == 2]
df5 = df[df['BatchSize'] == 5]
df10 = df[df['BatchSize'] == 10]
plt.figure()
plt.errorbar(df1['MSALength'], df1['AvTime5'], yerr=df1['StdTime5'],
             marker='.', linestyle='-', label='1')
plt.errorbar(df2['MSALength'], df2['AvTime5'], yerr=df2['StdTime5'],
             marker='.', linestyle='-', label='2')
plt.errorbar(df5['MSALength'], df5['AvTime5'], yerr=df5['StdTime5'],
             marker='.', linestyle='-', label='5')
plt.errorbar(df10['MSALength'], df10['AvTime5'], yerr=df10['StdTime5'],
             marker='.', linestyle='-', label='10')
plt.title('AlignScape execution time per epoch')
plt.xlabel('MSA length')
plt.ylabel('Average execution time per epoch (s)')
plt.legend(title='Batch size')
plt.grid(True)
plt.savefig('execution_time.pdf')
