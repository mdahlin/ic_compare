import numpy as np
from scipy.linalg import cholesky
from scipy import stats


def rank(X):
    m, n = X.shape
    res = np.empty((m, n), dtype=int)
    I = np.ogrid[:m, :n]
    rng, I[0] = I[0], X.argsort(axis=0)
    res[tuple(I)] = rng
    return res

def sum_Cerr(C, Ctar):
    """"correlation loss function to minimize"""
    return (
        abs(np.triu(C) - np.triu(Ctar)).sum()
    )

def ICreorder(X, Ctar):
    """Iman Conover reordering"""
    n_sim = X.shape[0]
    n_col = X.shape[1]
    # Apply a Cholesky decom to get an upper triangular matrix
    C = cholesky(Ctar)
    # create a vector of standard normal quantiles. Divide the quantiles so that their agg has std of 1.0
    a = np.arange(1, (n_sim+1))
    z = stats.norm.ppf(a/(n_sim+1))
    z = z/np.std(z, ddof=1)
    # copy this vector in score matrix M that has as many columns with normal scores as we want to generate random varaibles. randomly shuffle
    M = np.tile(z, (n_col, 1)).T # repmat(x,1,N)
    for j in range(0, M.shape[1]):
        p = np.random.permutation(n_sim)
        M[:, j] = M[p, j]
    # compute the correlation matrix E of this score matrix M
    E = np.dot(M.T, M)/n_sim
    # Apply cholesky decomp to get upper Cholesky triangle F
    F = cholesky(E)
    # Tie the matrices together via multiplication: T = M * inv(F) * C. This matrix has a targeted correlation structure
    IF = np.linalg.inv(F)
    TT = np.dot(np.dot(M, IF), C)
    # generate targeted rank correlation
    R = rank(TT)
    # get reordered simulations with desired correlation
    Y = np.copy(X)
    for j in range(0, n_col):
        Y[:, j] = X[R[:, j], j]
    return Y

    

def IC_IPCreorder(X, Ctar):
    """Iterated Perturbed Cholesky reordering after IC"""
    n_sim = X.shape[0]
    n_col = X.shape[1]
    # Apply a Cholesky decom to get an upper triangular matrix
    C = cholesky(Ctar)
    # create a vector of standard normal quantiles. Divide the quantiles so that their agg has std of 1.0
    a = np.arange(1, (n_sim+1))
    z = stats.norm.ppf(a/(n_sim+1))
    z = z/np.std(z, ddof=1)
    # copy this vector in score matrix M that has as many columns with normal scores as we want to generate random varaibles. randomly shuffle
    M = np.tile(z, (n_col, 1)).T # repmat(x,1,N)
    for j in range(0, M.shape[1]):
        p = np.random.permutation(n_sim)
        M[:, j] = M[p, j]
    # compute the correlation matrix E of this score matrix M
    E = np.dot(M.T, M)/n_sim
    # Apply cholesky decomp to get upper Cholesky triangle F
    F = cholesky(E)
    # Tie the matrices together via multiplication: T = M * inv(F) * C. This matrix has a targeted correlation structure
    IF = np.linalg.inv(F)
    TT = np.dot(np.dot(M, IF), C)
    # generate targeted rank correlation
    R = rank(TT)
    # get reordered simulations iwht desired correlation
    Y = np.copy(X)
    for j in range(0, n_col):
        Y[:, j] = X[R[:, j], j]

    # Initialize IPC after IC
    Cic = np.corrcoef(Y.T)
    Hic = cholesky(Cic) # from Iman Conover
    Hta = cholesky(Ctar)
    Hpr = np.copy(Hta) # perturb target with error
    ndx = ~np.eye(n_col, dtype=bool)
    err = Hpr[ndx]-Hic[ndx] # perturbed target - IC
    Hpr[ndx] = Hpr[ndx] + err
    E0 = 999
    E1 = sum_Cerr(C=Cic, Ctar=Ctar)

    # start iterative improvement
    while E1 < E0:
        #after perturbation, replace last IC steps
        TT = np.dot(np.dot(M, IF), Hpr)
        R = rank(TT)
        Yi = np.copy(X)
        for j in range(0, n_col):
            Yi[:, j] = X[R[:, j], j]
        #calc new correlation and corr error
        Cic = np.corrcoef(Yi.T)
        E0 = np.copy(E1)
        E1 = sum_Cerr(C=Cic, Ctar=Ctar)
        # calculate new Chol and Chol error
        Hic = cholesky(Cic)
        err = Hta[ndx] - Hic[ndx] # Original target - reordered
        Hpr[ndx] = Hpr[ndx] + err

    return Yi

def ILSreorder(Y, Ctar, n_iter=10000):
    """Iterated Local Search reordering"""
    n_sim = Y.shape[0]  # Yic is not going to be modified
    # initalization
    Yi = np.copy(Y)  # usually, preordered simlulations from IC
    C0 = np.corrcoef(Y.T)
    E0 = sum_Cerr(C0, Ctar=Ctar)
    # iterative optimziation
    for ii in range(1, n_iter+1):
        # identify problematic column with largest disrepencies
        Cerr = abs(C0 - Ctar)
        ndx_max = np.argmax(np.apply_along_axis(np.sum, 0, Cerr))
        # randomly select two components of xj to exhcnage and guarantee that we do not eveluate the same neighbor twice
        rng = np.random.default_rng()
        rnd_2ndx = rng.choice(n_sim, size=2, replace=False)
        Ycand = np.copy(Yi)
        Ycand[rnd_2ndx, ndx_max] = Ycand[np.flip(rnd_2ndx),ndx_max]
        #recalc actual corr and error
        C1 = np.corrcoef(Ycand.T)
        E1 = sum_Cerr(C1, Ctar=Ctar)
        # save reordered sims if error decreases
        if E1 <E0:
            Yi = np.copy(Ycand)
            E0 = E1
            C0 = C1

    return Yi

