import sys
sys.path.append('../')
import som_seq
import numpy as np
import quicksom
import seqdataloader as seqdataloader

epochs2 = np.arange(start=1,stop=21,step=2)
epochs5 = np.arange(start=24,stop=105,step=5)
epochs = np.concatenate((epochs2,epochs5),axis=None)
repetitions = 10
somside = 50
alpha = 0.5
batch_size = 10
#ali = '../data/Human_kinome/human_kinome_noPLK5_test.aln'
ali = '../data/Human_kinome/human_kinome_noPLK5.aln'

for epoch in epochs:
    f_out = open('/data/ifilella/test_errors_%d.csv'%epoch,'w')
    f_out.write('epochs,QE,TE\n')
    for i in range(repetitions):
        print(f'Epoch {epoch}, repetition {i}')
        som = som_seq.main(ali=ali,batch_size=batch_size,outname='/data/ifilella/test',somside=somside,nepochs=epoch,scheduler="exp",alpha=alpha,sigma=np.sqrt(somside*somside)/4.0)

        quantization_error = np.mean(som.error)

        dataset = seqdataloader.SeqDataset(ali)
        dataloader = quicksom.som.build_dataloader(dataset, num_workers=1, batch_size=batch_size,shuffle=False)
        bmu_indices = list()
        for i, (label, batch) in enumerate(dataloader):
            batch = batch.to(som.device)
            bmu_idx, error = som.inference_call(batch, n_bmu=2)
            bmu_idx = bmu_idx.cpu().detach().numpy()
            bmu_indices.append(bmu_idx)
        bmu_indices = np.concatenate(bmu_indices)
        bmu_indices = np.asarray(bmu_indices)
        topo_dists = np.array([float(som.distance_mat[int(first), int(second)].cpu().detach().numpy()) for first, second in bmu_indices])
        topo_error = np.sum(topo_dists > 1) / len(topo_dists)

        print(f'Quantization error: {quantization_error}')
        print(f'Topographical error: {topo_error}')
        f_out.write('%d,%f,%f\n'%(epoch,quantization_error,topo_error))
    f_out.close()
