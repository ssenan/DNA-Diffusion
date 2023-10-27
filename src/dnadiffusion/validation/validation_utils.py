import numpy as np
import pybedtools


def quantile_normalization(A):
    AA = np.zeros_like(A)
    I = np.argsort(A, axis=0)
    AA[I, np.arange(A.shape[1])] = np.mean(A[I, np.arange(A.shape[1])], axis=1)[:, np.newaxis]
    return AA


def extract_enhancer_sequence(genome_path: str, chr: str, start: str, end: str, shuffle: bool = False):
    a = pybedtools.BedTool(f"{chr} {start} {end}", from_string=True)
    a = a.sequence(fi=f"{genome_path}")
    sequence = open(a.seqfn).read().split("\n")[1]
    if shuffle:
        return "".join(np.random.permutation(list(sequence)))
    return sequence
