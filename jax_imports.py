import jax
import jax.numpy as jnp
aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')

# @jax.jit
def jax_score_matrix_vec(vec1, vec2, b62, gap_s=-5, gap_e=-1):
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
    final_scores = scores + max_gaps_aggregated * gap_s + max_exts_aggregated * gap_e
    return final_scores

def jax_rscore_matrix_vec(vec1, vec2, b62, gap_s=-5, gap_e=-1):
    """
    Jax Implementation
    """
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]
    nchars = len(aalist)
    rscore = jnp.shape(vec1)[1] * ((b62.sum())+nchars*(gap_s+gap_e))/(jnp.shape(b62)[0]*jnp.shape(b62)[1]+nchars*2)
    rscores = jnp.tile(rscore,(jnp.shape(vec1)[0],jnp.shape(vec2)[0]))
    return rscores

@jax.jit
def seqmetric_jax(seqs1, seqs2, b62):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    scores = jax_score_matrix_vec(seqs1, seqs2, b62=b62)
    #return -scores
    rscores = jax_rscore_matrix_vec(seqs1, seqs2, b62=b62)
    iscores = jax_score_matrix_vec(seqs1, seqs1, b62=b62)
    iscores = jnp.diagonal(iscores)
    iscores = jnp.repeat(iscores,(jnp.shape(seqs2)[0]))
    iscores = jnp.reshape(iscores, (jnp.shape(seqs1)[0],jnp.shape(seqs2)[0]))
    
    #Compute the B62 based distance
    denominators = iscores-rscores
    nominators = scores-rscores
    auxnominators = jnp.where(nominators < 0, 0.001, nominators)
    dists = (auxnominators)/(denominators)
    dists = -jnp.log(dists)*100
    return dists
