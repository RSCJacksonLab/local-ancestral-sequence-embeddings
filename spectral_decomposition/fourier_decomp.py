from ctypes import Array
import numpy as np

from numpy.typing import ArrayLike
from scipy import linalg, spatial
from sklearn.neighbors import NearestNeighbors

def get_adj_mat(rep_arr: ArrayLike):
    '''
    Get the adjacency matrix given a representation array.

    Arguments:
    ----------
    rep_arr : ArrayLike
        Representation array of size (n_seqs, rep_dim).
    '''
    # Make kNN
    dist_arr = spatial.distance_matrix(rep_arr, rep_arr)
    k = int(np.sqrt(len(rep_arr)))
    knn_fn = NearestNeighbors(
        n_neighbors=k,
        metric="precomputed"
    ).fit(dist_arr)
    # Determine graph Laplacian
    adj_mat = knn_fn.kneighbors_graph(dist_arr).toarray()
    ## remove self connections
    adj_mat -= np.eye(adj_mat.shape[0])
    ## make adjacency matrix symmetric
    adj_mat = adj_mat + adj_mat.T
    adj_mat[adj_mat > 1] = 1
    return adj_mat

def get_dirichlet_energy(
    adj_mat: ArrayLike, 
    y: ArrayLike, 
    return_lacplacian=False
):
    '''
    Get the normalized Dirichlet energy given the adjacency matrix
    array and signals (y).
    
    Arguments:
    ----------
    adj_mat : ArrayLike
        Array of (n_seqs, n_seqs) containing ones where sequences are
        neighboring and zeros otherwise.
    y : ArrayLike
        Array of (n_seqs,) containing signals for each sequence - 
        where signals are fitness values.
    return_laplacian : bool, default `False`
        If "True" return the normalized Dirichlet Energy and the
        Laplacian as a Tuple. Else, return the normalized Dirichlet
        Energy alone.
    '''
    # make diagonal matrix from adjacency
    diag_mat = np.diag(np.sum(adj_mat, axis=0))
    # determine graph laplacian
    laplacian = diag_mat - adj_mat
    # determine dirichlet energy
    dir_en = (y @ laplacian) @ y.T
    norm_dir_en = dir_en/len(y)
    if return_lacplacian:
        return norm_dir_en, laplacian
    else:
        return norm_dir_en

def get_global_dirichlet_energy(rep_arr: ArrayLike, y: ArrayLike):
    '''
    Get the normalised Dirichlet energy for the representation space
    given the representation array and signals (y).

    Arguments:
    ----------
    rep_arr : ArrayLike
        Representation array of size (n_seqs, rep_dim).
    y : ArrayLike
        Array of (n_seqs,) containing signals for each sequence - 
        where signals are fitness values.
    '''
    adj_mat = get_adj_mat(rep_arr)
    dir_en = get_dirichlet_energy(adj_mat, y)
    return dir_en

def get_local_dirichlet_energy(rep_arr: ArrayLike, y: ArrayLike):
    '''
    Get the normalised Dirichlet energy for each datapoint given the
    representation array and signals (y). 
    
    Dirichlet energy must be normalised despite the use of a kNN as to
    produce a symmetric adjacency matrix, the directed graph must be
    converted into an undirected graph where the degree of each node
    may be greater than k.

    Arguments:
    ----------
    rep_arr : ArrayLike
        Representation array of size (n_seqs, rep_dim).
    y : ArrayLike
        Array of (n_seqs,) containing signals for each sequence - 
        where signals are fitness values.
    '''
    adj_mat = get_adj_mat(rep_arr)
    local_en_ls = []
    for i in range(len(adj_mat)):
        # determine neighbours of the ith node
        neighbour_idx = np.where(adj_mat[i] == 1)
        # plus one to account for self
        n = np.sum(adj_mat[i] == 1) + 1
        # make local adj mat
        local_adj_mat = np.zeros((n, n))
        local_adj_mat[:, 0] = 1
        local_adj_mat[0, :] = 1
        local_adj_mat[0, 0] = 0
        # get local signals
        local_y = np.concatenate(
            (y[i].reshape(1, -1), y[neighbour_idx].reshape(1, -1)), 
            axis=1
        )[0]
        local_dir_en = get_dirichlet_energy(local_adj_mat, local_y)
        local_en_ls.append(local_dir_en)
    return local_en_ls

def graph_fourier_decomposition(rep_arr: ArrayLike, y: ArrayLike):
    '''
    Perform Graph Fourier Transfomr over graph made from provided
    representation array and signals.

    Arguments:
    ----------
    rep_arr : ArrayLike
        Representation array of size (n_seqs, rep_dim).
    y : ArrayLike
        Array of (n_seqs,) containing signals for each sequence - 
        where signals are fitness values.
    '''
    adj_mat = get_adj_mat(rep_arr)
    _, laplacian = get_dirichlet_energy(adj_mat, y, return_lacplacian=True)
    eigenvals, eigenvecs = linalg.eigh(laplacian)
    gft = eigenvecs.T * y
    return eigenvals, eigenvecs, gft