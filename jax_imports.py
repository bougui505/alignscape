import jax
import jax.numpy as jnp

# @jax.jit
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
    vmax = jax.vmap(jnp.maximum, in_axes=(0, None))
    max_gaps = vmax(gaps1, gaps2)
    max_gaps_aggregated = max_gaps.sum(axis=2)
    max_exts = vmax(exts1, exts2)
    max_exts_aggregated = max_exts.sum(axis=2)
    final_scores = scores + max_gaps_aggregated * -5 + max_exts_aggregated * -1
    return final_scores


@jax.jit
def seqmetric_jax(seqs1, seqs2, b62):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    scores = jax_score_matrix_vec(seqs1, seqs2, b62=b62)
    return -scores