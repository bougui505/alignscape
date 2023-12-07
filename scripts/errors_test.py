import numpy as np
from alignscape import quicksom
from alignscape import som_seq
from alignscape.utils import seqdataloader

epochs2 = np.arange(start=1,stop=21,step=2)
epochs5 = np.arange(start=24,stop=105,step=5)
epochs = np.concatenate((epochs2,epochs5),axis=None)
repetitions = 10
somside = 50
alpha = 0.5
batch_size = 8
#ali = '../data/Human_kinome/human_kinome_noPLK5.aln'
#ali = '../data/Human_gpcr/gpcrs_human_gapyness09.aln'
ali = '../data/T6SS/TssB/TssB_rmout_gapyness090.fasta'
for epoch in epochs[15:]:
    f_out = open('/data/ifilella/TssB_errors_%d.csv'%epoch,'w')
    f_out.write('epochs,QE,TE\n')
    for i in range(repetitions):
        print(f'Epoch {epoch}, repetition {i}')
        somobj = som_seq.main(ali=ali,batch_size=batch_size,outname='/data/ifilella/test',somside=somside,nepochs=epoch,scheduler="exp",alpha=alpha,sigma=np.sqrt(somside*somside)/4.0)

        quantization_error = np.mean(somobj.error)

        dataset = seqdataloader.SeqDataset(ali)
        dataloader = quicksom.somobj.build_dataloader(dataset, num_workers=1, batch_size=batch_size,shuffle=False)
        bmu_indices = list()
        for i, (label, batch) in enumerate(dataloader):
            batch = batch.to(somobj.device)
            bmu_idx, error = somobj.inference_call(batch, n_bmu=2)
            bmu_idx = bmu_idx.cpu().detach().numpy()
            bmu_indices.append(bmu_idx)
        bmu_indices = np.concatenate(bmu_indices)
        bmu_indices = np.asarray(bmu_indices)
        topo_dists = np.array([float(somobj.distance_mat[int(first), int(second)].cpu().detach().numpy()) for first, second in bmu_indices])
        topo_error = np.sum(topo_dists > 1) / len(topo_dists)

        print(f'Quantization error: {quantization_error}')
        print(f'Topographical error: {topo_error}')
        f_out.write('%d,%f,%f\n'%(epoch,quantization_error,topo_error))
    f_out.close()
