
# AlignScape

The AlignScape package takes an MSA as input and uses a self-organizing map to compute the sequence landscape of the input data.

![GitFigure](https://github.com/bougui505/alignscape/assets/27772386/39490a0b-8802-4ec1-9fcc-8bbee90a1fca)

AlignScape can be utilized as a Jupyter Notebook on their local machines or within a Google Colaboratory Notebook, supporting both CPUs and GPUs.

[AlignScape Google Colab](https://github.com/bougui505/alignscape/blob/master/alignscape.ipynb)

## Using alignscape locally using a Singularity/Apptainer image

### Download the image `alignscape.sif`
The image can be downloaded using the following link on Zenodo platform:

https://zenodo.org/records/10417520

### Using the Singularity image
The image can be used as follow:

```
singularity run --nv alignscape.sif align_scape -h
```

If you don't have a GPU on your computer removed the `--nv` option.

## Installation

To install AlignScape localy:

```bash
  git clone https://github.com/bougui505/alignscape
  cd alignscape
  git clone https://github.com/bougui505/quicksom
```
    
## Tests

To check AlignScape local installation run the following commands

```bash
./test_som.sh
python test_postprocess.py
```


## Usage/Examples

Compute the SOM
```bash
python -m alignscape.align_scape -a alignment.fasta -b 10 --somside 90 --alpha 0.5 --nepochs 200 -o outname
```

Plot the U-matrix
```bash
python -m alignscape.plot_umat -s somfile -o outname -d delimiter 
```


## Reference

Please cite ...

