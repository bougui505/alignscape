
# AlignScape

The AlignScape package takes an MSA as input and uses a self-organizing map to compute the sequence landscape of the input data.

![GitFigure](https://github.com/bougui505/alignscape/assets/27772386/39490a0b-8802-4ec1-9fcc-8bbee90a1fca)

AlignScape can be utilized as a Jupyter Notebook on their local machines or within a Google Colaboratory Notebook, supporting both CPUs and GPUs.

[AlignScape Google Colab](https://github.com/bougui505/alignscape/blob/master/alignscape.ipynb)

## AlignScape locally with a Singularity/Apptainer image

### Download the image `alignscape.sif`
The image can be downloaded using the following link on Zenodo platform:

https://zenodo.org/records/10417520

### Using the Singularity image
The image can be used as follows:

```
singularity run --nv alignscape.sif align_scape -h
```

If you don't have a GPU on your computer, remove the `--nv` option.

To run AlignScape on a alignment in fasta format, simply run:

```
singularity run --nv alignscape.sif align_scape -a alignment.fasta --nepochs 100
```

PLEASE NOTE: Singularity mounts only a part of your file system. To ensure that your fasta file is accessible within the singularity image, you should place it in your home directory or in your home directory hierarchy. This should guarantee its availability for use.

Plotting the U-matrix
```
singularity run --nv alignscape.sif python3 /usr/local/lib/python3.11/dist-packages/alignscape/plot_umat.py -s somfile -o outname
```

## AlignScape locally within a conda environment

### Create a conda enviroment and install AlignScape on it:

```bash
conda create -n alignscape python=3.12
ENV_PATH=$(conda info --envs | grep 'alignscape' | awk '{print $NF}')
cd $ENV_PATH/lib/python3.12/site-packages/
git clone https://github.com/bougui505/alignscape
cd alignscape
git clone https://github.com/bougui505/quicksom
```

A list of the needed dependencies can be found at `apptainer/alignscape.def`

### Using AlignScape in the conda environment:

```
conda activate alignscape
```

To run AlignScape on a alignment in fasta format, simply run

```
python -m alignscape.align_scape -a alignment.fasta
```

Plotting the U-matrix

```
python -m alignscape.plot_umat -s somfile -o outname
```

## Test

To check AlignScape local installation:

```bash
./test_som.sh
```


## Reference

Please cite ...

