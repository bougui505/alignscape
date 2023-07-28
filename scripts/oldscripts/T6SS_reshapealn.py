import glob
import sys
sys.path.append('../')
import fastaf
import ast
import os

fsubtypes = '/home/ifilella/Projects/quicksom_seq/data/T6SS/subtypes.txt'
f = open(fsubtypes, 'r')
contents = f.read()
dsubtypes = ast.literal_eval(contents)
f.close()
prot = 'VgrG'
directory = '/home/ifilella/Projects/quicksom_seq/data/T6SS/' + prot
print(dsubtypes)
exit()
queries = []
alns = glob.glob(directory+'/*.aln')
for aln in alns:
    print(aln)
    ali = fastaf.Alignment(aln)
    for title in ali.names:
        if prot in title:
            newtitle = title.replace("-"+prot,"")
            bacteria = newtitle.split('|')[0]
            subtype = dsubtypes[bacteria]
            newtitle = subtype + '_' + newtitle
            if  (prot + '.aln') in aln:
                queries.append('>'+newtitle)
        else:
            newtitle = 'unk_' + title
            #print(newtitle)
        ali.ali[newtitle] = ali.ali[title]
        del ali.ali[title]
    ali.write_ali(aln.replace('.aln','.fasta'))

f = open(directory + '/' + prot + '_queries_fasta.txt','w')
for query in queries:
    f.write(query+'\n')
f.close()
