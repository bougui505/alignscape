import sys
import numpy as np
import torch

try:
    import functorch

    FUNCTORCH_AVAIL = True
except ImportError:
    print("Running without functorch, please install it")
    FUNCTORCH_AVAIL = False
import time
import re
import functools

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import device_put

blosum62 = {
    ("W", "F"): 1, ("L", "R"): -2, ("S", "P"): -1, ("V", "T"): 0,
    ("Q", "Q"): 5, ("N", "A"): -2, ("Z", "Y"): -2, ("W", "R"): -3,
    ("Q", "A"): -1, ("S", "D"): 0, ("H", "H"): 8, ("S", "H"): -1,
    ("H", "D"): -1, ("L", "N"): -3, ("W", "A"): -3, ("Y", "M"): -1,
    ("G", "R"): -2, ("Y", "I"): -1, ("Y", "E"): -2, ("B", "Y"): -3,
    ("Y", "A"): -2, ("V", "D"): -3, ("B", "S"): 0, ("Y", "Y"): 7,
    ("G", "N"): 0, ("E", "C"): -4, ("Y", "Q"): -1, ("Z", "Z"): 4,
    ("V", "A"): 0, ("C", "C"): 9, ("M", "R"): -1, ("V", "E"): -2,
    ("T", "N"): 0, ("P", "P"): 7, ("V", "I"): 3, ("V", "S"): -2,
    ("Z", "P"): -1, ("V", "M"): 1, ("T", "F"): -2, ("V", "Q"): -2,
    ("K", "K"): 5, ("P", "D"): -1, ("I", "H"): -3, ("I", "D"): -3,
    ("T", "R"): -1, ("P", "L"): -3, ("K", "G"): -2, ("M", "N"): -2,
    ("P", "H"): -2, ("F", "Q"): -3, ("Z", "G"): -2, ("X", "L"): -1,
    ("T", "M"): -1, ("Z", "C"): -3, ("X", "H"): -1, ("D", "R"): -2,
    ("B", "W"): -4, ("X", "D"): -1, ("Z", "K"): 1, ("F", "A"): -2,
    ("Z", "W"): -3, ("F", "E"): -3, ("D", "N"): 1, ("B", "K"): 0,
    ("X", "X"): -1, ("F", "I"): 0, ("B", "G"): -1, ("X", "T"): 0,
    ("F", "M"): 0, ("B", "C"): -3, ("Z", "I"): -3, ("Z", "V"): -2,
    ("S", "S"): 4, ("L", "Q"): -2, ("W", "E"): -3, ("Q", "R"): 1,
    ("N", "N"): 6, ("W", "M"): -1, ("Q", "C"): -3, ("W", "I"): -3,
    ("S", "C"): -1, ("L", "A"): -1, ("S", "G"): 0, ("L", "E"): -3,
    ("W", "Q"): -2, ("H", "G"): -2, ("S", "K"): 0, ("Q", "N"): 0,
    ("N", "R"): 0, ("H", "C"): -3, ("Y", "N"): -2, ("G", "Q"): -2,
    ("Y", "F"): 3, ("C", "A"): 0, ("V", "L"): 1, ("G", "E"): -2,
    ("G", "A"): 0, ("K", "R"): 2, ("E", "D"): 2, ("Y", "R"): -2,
    ("M", "Q"): 0, ("T", "I"): -1, ("C", "D"): -3, ("V", "F"): -1,
    ("T", "A"): 0, ("T", "P"): -1, ("B", "P"): -2, ("T", "E"): -1,
    ("V", "N"): -3, ("P", "G"): -2, ("M", "A"): -1, ("K", "H"): -1,
    ("V", "R"): -3, ("P", "C"): -3, ("M", "E"): -2, ("K", "L"): -2,
    ("V", "V"): 4, ("M", "I"): 1, ("T", "Q"): -1, ("I", "G"): -4,
    ("P", "K"): -1, ("M", "M"): 5, ("K", "D"): -1, ("I", "C"): -1,
    ("Z", "D"): 1, ("F", "R"): -3, ("X", "K"): -1, ("Q", "D"): 0,
    ("X", "G"): -1, ("Z", "L"): -3, ("X", "C"): -2, ("Z", "H"): 0,
    ("B", "L"): -4, ("B", "H"): 0, ("F", "F"): 6, ("X", "W"): -2,
    ("B", "D"): 4, ("D", "A"): -2, ("S", "L"): -2, ("X", "S"): 0,
    ("F", "N"): -3, ("S", "R"): -1, ("W", "D"): -4, ("V", "Y"): -1,
    ("W", "L"): -2, ("H", "R"): 0, ("W", "H"): -2, ("H", "N"): 1,
    ("W", "T"): -2, ("T", "T"): 5, ("S", "F"): -2, ("W", "P"): -4,
    ("L", "D"): -4, ("B", "I"): -3, ("L", "H"): -3, ("S", "N"): 1,
    ("B", "T"): -1, ("L", "L"): 4, ("Y", "K"): -2, ("E", "Q"): 2,
    ("Y", "G"): -3, ("Z", "S"): 0, ("Y", "C"): -2, ("G", "D"): -1,
    ("B", "V"): -3, ("E", "A"): -1, ("Y", "W"): 2, ("E", "E"): 5,
    ("Y", "S"): -2, ("C", "N"): -3, ("V", "C"): -1, ("T", "H"): -2,
    ("P", "R"): -2, ("V", "G"): -3, ("T", "L"): -1, ("V", "K"): -2,
    ("K", "Q"): 1, ("R", "A"): -1, ("I", "R"): -3, ("T", "D"): -1,
    ("P", "F"): -4, ("I", "N"): -3, ("K", "I"): -3, ("M", "D"): -3,
    ("V", "W"): -3, ("W", "W"): 11, ("M", "H"): -2, ("P", "N"): -2,
    ("K", "A"): -1, ("M", "L"): 2, ("K", "E"): 1, ("Z", "E"): 4,
    ("X", "N"): -1, ("Z", "A"): -1, ("Z", "M"): -1, ("X", "F"): -1,
    ("K", "C"): -3, ("B", "Q"): 0, ("X", "B"): -1, ("B", "M"): -3,
    ("F", "C"): -2, ("Z", "Q"): 3, ("X", "Z"): -1, ("F", "G"): -3,
    ("B", "E"): 1, ("X", "V"): -1, ("F", "K"): -3, ("B", "A"): -2,
    ("X", "R"): -1, ("D", "D"): 6, ("W", "G"): -2, ("Z", "F"): -3,
    ("S", "Q"): 0, ("W", "C"): -2, ("W", "K"): -3, ("H", "Q"): 0,
    ("L", "C"): -1, ("W", "N"): -4, ("S", "A"): 1, ("L", "G"): -4,
    ("W", "S"): -3, ("S", "E"): 0, ("H", "E"): 0, ("S", "I"): -2,
    ("H", "A"): -2, ("S", "M"): -1, ("Y", "L"): -1, ("Y", "H"): 2,
    ("Y", "D"): -3, ("E", "R"): 0, ("X", "P"): -2, ("G", "G"): 6,
    ("G", "C"): -3, ("E", "N"): 0, ("Y", "T"): -2, ("Y", "P"): -3,
    ("T", "K"): -1, ("A", "A"): 4, ("P", "Q"): -1, ("T", "C"): -1,
    ("V", "H"): -3, ("T", "G"): -2, ("I", "Q"): -3, ("Z", "T"): -1,
    ("C", "R"): -3, ("V", "P"): -2, ("P", "E"): -1, ("M", "C"): -1,
    ("K", "N"): 0, ("I", "I"): 4, ("P", "A"): -1, ("M", "G"): -3,
    ("T", "S"): 1, ("I", "E"): -3, ("P", "M"): -2, ("M", "K"): -1,
    ("I", "A"): -1, ("P", "I"): -3, ("R", "R"): 5, ("X", "M"): -1,
    ("L", "I"): 2, ("X", "I"): -1, ("Z", "B"): 1, ("X", "E"): -1,
    ("Z", "N"): 0, ("X", "A"): 0, ("B", "R"): -1, ("B", "N"): 3,
    ("F", "D"): -3, ("X", "Y"): -1, ("Z", "R"): 0, ("F", "H"): -1,
    ("B", "F"): -3, ("F", "L"): 0, ("X", "Q"): -1, ("B", "B"): 4
}


def get_blosum62():
    aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')
    b62 = np.zeros((23, 23))
    for k in blosum62:
        i0 = aalist.index(k[0])
        i1 = aalist.index(k[1])
        b62[i0, i1] = blosum62[k]
        b62[i1, i0] = blosum62[k]
    return b62


def _substitute_opening_gap_char(seq):
    rex = re.compile('[A-Z]-')
    newseq = list(seq)
    if newseq[0] == "-":
        newseq[0] = "|"
    iterator = rex.finditer(seq)
    for match in iterator:
        try:
            newseq[match.span()[1] - 1] = "|"
        except:
            continue
    return "".join(newseq)


def seq2vec(sequence, dtype='prot'):
    """
    - sequence: string
    """
    aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')
    nucllist = list('ATGCSWRYKMBVHDN|-')
    if dtype == 'prot':
        mapper = dict([(r, i) for i, r in enumerate(aalist)])
        naa_types = len(aalist)
    elif dtype == 'nucl':
        mapper = dict([(r, i) for i, r in enumerate(nucllist)])
        naa_types = len(nucllist)
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    sequence = _substitute_opening_gap_char(sequence)
    naa = len(sequence)
    vec = np.zeros((naa, naa_types))
    for i, res in enumerate(list(sequence)):
        ind = mapper[res]
        vec[i, ind] = 1.
    return vec


def vectorize(sequences, dtype='prot'):
    vectors = np.asarray([seq2vec(s, dtype).flatten() for s in sequences])
    return vectors


def read_fasta(fastafilename, names=None):
    """
    """
    sequences = []
    seq = None
    seqname = None
    seqnames = []
    with open(fastafilename) as fastafile:
        for line in fastafile:
            if line[0] == ">":  # sequence name
                if seq is not None:
                    if names is None or seqname in names:
                        seqnames.append(seqname)
                        sequences.append(seq)
                seqname = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
    if names is None or seqname in names:
        sequences.append(seq)
        seqnames.append(seqname)
    return seqnames, sequences


def torchify(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def torch_score_novmap(vec1, vec2, dtype="prot", gap_s=-5, gap_e=-1, b62=None, NUC44=None):
    """
    PyTorch Implementation, the old way
    """
    if dtype == 'prot':
        matrix = b62
    elif dtype == 'nucl':
        matrix = NUC44
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    device1 = vec1.device.type
    device2 = matrix.device.type
    device3 = vec2.device.type
    devices = [device1, device2, device3]
    if 'cuda' in devices and 'cpu' in devices:
        vec1 = torchify(vec1)
        vec2 = torchify(vec2)
        matrix = torchify(matrix)
    vec1 = vec1.float()
    vec2 = vec2.float()
    matrix = matrix.float()
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]
    matv2 = torch.matmul(matrix[None, ...], torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)
    for i in range(len(vec1)):
        # scores.shape = (a, b) with a: size of batch and b size of SOM
        scores[i] += torch.maximum(vec1[i, ..., -2], vec2[..., -2]).sum(axis=1) * gap_s
        scores[i] += torch.maximum(vec1[i, ..., -1], vec2[..., -1]).sum(axis=1) * gap_e
    # scores = list(scores.to('cpu').numpy())
    if 'cpu' in devices:
        scores = scores.to('cpu')
    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def torch_score_matrix_vec(vec1, vec2, dtype="prot", gap_s=-5, gap_e=-1, b62=None, NUC44=None):
    """
    PyTorch Implementation, trying to vectorize the maximum application.
    This is ported in the main now, and is sometimes giving a good speedup, sometimes not.
    """
    if dtype == 'prot':
        matrix = b62
    elif dtype == 'nucl':
        matrix = NUC44
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")

    a = time.time()
    vec1 = vec1.float()
    vec2 = vec2.float()
    matrix = matrix.float()
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]
    matv2 = torch.matmul(matrix[None, ...], torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)

    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    # In terms of timing, the slow steps remains the max application.
    # For certain batch sizes, it gives a good speedup, for others not.
    # Applying the max is by far the bottlenecks.
    if FUNCTORCH_AVAIL:
        # print()
        # print(gaps1.shape)
        # print(gaps2.shape)
        # print(exts1.shape)
        # print(exts2.shape)

        vmax = functorch.vmap(torch.maximum, in_dims=(0, None))
        max_gaps = vmax(gaps1, gaps2)
        max_gaps_aggregated = max_gaps.sum(axis=2)
        max_exts = vmax(exts1, exts2)
        max_exts_aggregated = max_exts.sum(axis=2)
        scores = scores + max_gaps_aggregated * gap_s + max_exts_aggregated * gap_e
    else:
        for i in range(len(vec1)):
            # scores.shape = (a, b) with a: size of batch and b size of SOM
            scores[i] += torch.maximum(gaps1[i, ...], gaps2).sum(axis=1) * gap_s
            scores[i] += torch.maximum(exts1[i, ...], exts2).sum(axis=1) * gap_e

    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def torch_score_vmap(vec1, vec2, dtype="prot", gap_s=-5, gap_e=-1, b62=None, NUC44=None):
    """
    PyTorch Implementation
    """
    if dtype == 'prot':
        matrix = b62
    elif dtype == 'nucl':
        matrix = NUC44
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    vec1 = vec1.float()
    vec2 = vec2.float()
    matrix = matrix.float()
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]

    print(vec1.shape)
    print(vec2.shape)
    print(b62.shape)
    sys.exit()

    matv2 = torch.matmul(matrix[None, ...], torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)
    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    vmax = functorch.vmap(torch.maximum, in_dims=(0, None))
    max_gaps = vmax(gaps1, gaps2)
    max_gaps_aggregated = max_gaps.sum(axis=2)
    max_exts = vmax(exts1, exts2)
    max_exts_aggregated = max_exts.sum(axis=2)
    scores = scores + max_gaps_aggregated * gap_s + max_exts_aggregated * gap_e
    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def to_compile_torch_score_matrix_vec(vec1, vec2, b62):
    """
    No arguments and branching for compilation with tracing. In any case it does not yield improvements
    """
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]
    matv2 = torch.matmul(b62[None, ...], torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)
    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    vmax = functorch.vmap(torch.maximum, in_dims=(0, None))
    max_gaps = vmax(gaps1, gaps2)
    max_gaps_aggregated = max_gaps.sum(axis=2)
    max_exts = vmax(exts1, exts2)
    max_exts_aggregated = max_exts.sum(axis=2)
    scores = scores + max_gaps_aggregated * -5 + max_exts_aggregated * -1
    return scores


def jax_score_matrix_vec(vec1, vec2, b62):
    """
    Jax Implementation
    """
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]

    matv2 = jnp.matmul(b62[None, ...], jnp.swapaxes(vec2[..., :-2], 1, 2))
    scores = jnp.einsum('aij,bji->ab', vec1[..., :-2], matv2)
    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    vmax = vmap(jnp.maximum, in_axes=(0, None))
    max_gaps = vmax(gaps1, gaps2)
    max_gaps_aggregated = max_gaps.sum(axis=2)
    max_exts = vmax(exts1, exts2)
    max_exts_aggregated = max_exts.sum(axis=2)
    final_scores = scores + max_gaps_aggregated * -5 + max_exts_aggregated * -1
    return final_scores


def shape_seq(seqs1, seqs2):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    return seqs1, seqs2


def torch_test(torch_batch_vecs, torch_centroid_vecs, torch_b62):
    n_reps = 10
    assert n_reps >= 1

    # There is a weird phenomenon that makes this first loop much slower than the rest on GPU (probs kernel loading)
    # torch.Size([100, 203, 25])
    # torch.Size([2500, 203, 25])
    # --- On my CPU ---
    # vec time :  0.7957186698913574
    # native time :  1.310312271118164
    # native time double check:  1.2960045337677002
    # vec time double check:  0.7912266254425049
    # --- On the GPU ---
    # vec time :  0.36736464500427246
    # native time :  0.05288386344909668
    # native time double check:  0.05283069610595703
    # vec time double check:  0.04572415351867676

    global FUNCTORCH_AVAIL
    a = time.time()
    for i in range(n_reps):
        vec_scores = torch_score_matrix_vec(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('vec time : ', (time.time() - a) / n_reps)

    a = time.time()
    for i in range(n_reps):
        scores = torch_score_novmap(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('no vmap time : ', (time.time() - a) / n_reps)

    a = time.time()
    for i in range(n_reps):
        vec_scores = torch_score_vmap(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('vmap time: ', (time.time() - a) / n_reps)

    FUNCTORCH_AVAIL = False
    a = time.time()
    for i in range(n_reps):
        vec_scores = torch_score_matrix_vec(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('no vmap double check: ', (time.time() - a) / n_reps)

    FUNCTORCH_AVAIL = True
    a = time.time()
    for i in range(n_reps):
        vec_scores = torch_score_matrix_vec(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('vmap double check: ', (time.time() - a) / n_reps)

    # TODO : This result is very surprising !
    #  The matrices have the same shape as in the SOM, there are on the GPU exactly the same
    #  The distance code is a copy from the one run by alignscape.
    #  However, on the gpu, functorch is MUCH faster for the som (0.0110 vs 0.00075)
    #    and approx the same here (native time double check:  0.0168, vec time triple check:  0.0151)
    #  I don't get it !

    sys.exit()

    a = time.time()
    traced = torch.jit.trace(to_compile_torch_score_matrix_vec, (torch_batch_vecs, torch_centroid_vecs, torch_b62))
    print('time to trace : ', time.time() - a)
    a = time.time()
    for i in range(n_reps):
        traced_scores = traced(torch_batch_vecs, torch_centroid_vecs, b62=torch_b62)
    print((traced_scores - scores).mean())
    print('traced time : ', time.time() - a)
    print((vec_scores - scores).mean())


def jax_test(jax_score_matrix_vec=jax_score_matrix_vec):
    # Make them torch and put on device
    jax_b62 = device_put(B62)
    jax_inputvectors = device_put(inputvectors)
    jax_targets = device_put(centroid_vecs)

    # torch_b62 = torchify(B62)
    # torch_inputvectors = torchify(inputvectors)
    # torch_targets = torchify(centroid_vecs)
    # torch_scores = torch_score_matrix_vec(torch_inputvectors, torch_targets, b62=torch_b62)
    # jax_scores = jax_score_matrix_vec(jax_inputvectors, jax_targets, jax_b62)

    # No jit, just jax
    a = time.time()
    for i in range(2):
        jax_scores = jax_score_matrix_vec(jax_inputvectors, jax_targets, jax_b62)
    print('native time : ', time.time() - a)

    a = time.time()
    jax_score_matrix_vec = jax.jit(jax_score_matrix_vec)
    print('time to jit : ', time.time() - a)

    a = time.time()
    for i in range(2):
        scores = jax_score_matrix_vec(jax_inputvectors, jax_targets, b62=jax_b62)
    print('traced time : ', time.time() - a)

    a = time.time()
    for i in range(2):
        torch_scores = torch_score_matrix_vec(torch_inputvectors, torch_targets, b62=torch_b62)
    print('torch time : ', time.time() - a)


if __name__ == '__main__':
    # As in fitting a SOM, select the right amount of sequences
    seqnames, sequences = read_fasta('data/TssB.aln')
    seqnames = np.asarray(seqnames)
    inputvectors = vectorize(sequences, dtype='prot')
    batch_vecs = np.copy(inputvectors)[:100]
    centroid_vecs = np.copy(inputvectors)[:2500]

    # Reshape them and send them to torch
    # (100, 1, 5075) -> (100, 203, 25)
    # (2500, 5075)   -> (2500, 203, 25)
    nchar = 25
    batch_size = batch_vecs.shape[0]
    seqlenght = batch_vecs.shape[-1] // nchar
    n2 = centroid_vecs.shape[0]
    batch_vecs = batch_vecs.reshape((batch_size, seqlenght, nchar))
    centroid_vecs = centroid_vecs.reshape((n2, seqlenght, nchar))
    B62 = get_blosum62()
    torch_batch_vecs = torchify(batch_vecs)
    torch_centroid_vecs = torchify(centroid_vecs)
    torch_b62 = torchify(B62)

    # Make a Pytorch competition !
    # torch_test(torch_batch_vecs=torch_batch_vecs, torch_centroid_vecs=torch_centroid_vecs, torch_b62=torch_b62)
    jax_test()
