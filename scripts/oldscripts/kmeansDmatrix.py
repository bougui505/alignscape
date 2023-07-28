import dmatrix
import argparse

def main(sys,labels,subtypes,epochs,deli):
    for label in labels:
        print(label)
        DMcomplete = dmatrix.Dmatrix(load='%s_dist/%s/%s_%se_dmatrix.p'%(sys,label,label,epochs),subtypes=subtypes,delimiter=deli)
        for i in range(2):
            DMkmeans = dmatrix.Dmatrix(load='%s_dist/%s/%s_2kmeans%d_%se_dmatrix.p'%(sys,label,label,i,epochs),subtypes=subtypes,delimiter=deli)
            DMcomplete.update(DMkmeans)
        out = '%s_dist/%s/%s_2kmeans_%se_dmatrix'%(sys,label,label,epochs)
        DMcomplete.save(out=out)
        DMcomplete.plot(out=out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-sys', '--sys', help = 'System, only T6SS and T2SS are allowed', required = True)
    parser.add_argument('-ep', '--epochs', help = 'Number of epochs employed to generate the SOM which generate the dmatrices', required = True, type=str)
    parser.add_argument('--deli',help = 'Delimiter to trim the queries tittles',default = '-', type = str)
    args = parser.parse_args()
    sys = args.sys
    deli = args.deli
    epochs = args.epochs

    if sys != 'T6SS' and sys != 'T2SS':
        raise KeyError('System must be T6SS or T2SS')
    else:
        if sys == 'T6SS':
            labels = ['TssA','TssB','TssC','TssE','TssF','TssG','TssK','TssL','TssM','ClpV','VgrG','hcp','TssJ']
            subtypes = 'data/T6SS/subtypes.txt'
        elif sys == 'T2SS':
            labels = ['C','D','E','F','G','H','I','J','K','L','M','O']
            subtypes = None

    main(sys,labels,subtypes,epochs,deli)
