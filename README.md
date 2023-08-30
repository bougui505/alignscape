
# SOMseq

The SOMseq package takes an MSA as input and uses a self-organizing map to compute the sequence landscape of the input data.

![GitFigure](https://github.com/bougui505/quicksom_seq/assets/27772386/39490a0b-8802-4ec1-9fcc-8bbee90a1fca)

SOMseq can be utilized as a Jupyter Notebook on their local machines or within a Google Colaboratory Notebook, supporting both CPUs and GPUs.

[SOMseq Google Colab](https://colab.research.google.com/github/bougui505/quicksom_seq/blob/master/somseq.ipynb)



## Installation

To install SOMseq localy:

```bash
  git clone https://github.com/bougui505/quicksom_seq
  cd quicksom_seq
  git clone https://github.com/bougui505/quicksom
```
    
## Tests

To check SOMseq local installation run the following commands

```bash
./test_som.sh
python test_postprocess.py
```


## Usage/Examples

Compute the SOM
```bash
python -m quicksom_seq.som_seq -a alignment.fasta -b 10 --somside 90 --alpha 0.5 --nepochs 200 -o outname
```

Plot the U-matrix
```bash
python -m quicksom_seq.plot_umat -s somfile -o outname -d delimiter 
```


## Reference

Please cite ...

