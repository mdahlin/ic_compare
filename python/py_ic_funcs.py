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
    mod = np.triu(C).flatten()
    tru = np.triu(Ctar).flatten()
    ndx = ~np.in1d(mod, [0, 1])
    mod = mod[ndx]
    tru = tru[ndx]
    return np.sum(abs(tru - mod))

def ICreorder(X, Ctar):
    """Iman Conover reordering"""
    n_sim = X.shape[0]
    n_col = X.shape[1]
    X.sort(axis=0)
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
    return Y

    

def IC_IPCreorder(X, Ctar):
    """Iterated Perturbed Cholesky reordering after IC"""
    n_sim = X.shape[0]
    n_col = X.shape[1]
    X.sort(axis=0)
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
    Hpr = np.copy(cholesky(Ctar)) # perturb target with error
    breakpoint()
    ndx = ~np.eye(n_col, dtype=bool)
    err = Hpr[ndx]-Hic[ndx] # perturbed target - IC
    Hpr[ndx] = Hpr[ndx] + err
    E0 = 999
    E1 = sum_Cerr(C=Cic, Ctar=S) # Ctar? or separate variable?
    Serr_iter = sum_Cerr(C=Cic, Ctar=S) # Ctar? or separate variable?

    # start iterative improvement
    while E1 < R0:
        #after perturbation, replace last IC steps
        TT = np.dot(np.dot(M, IF), Hrp)
        R = rank(TT)
        Yi = np.copy(X)
        for j in range(0, n_col):
            Yi[:, j] = X[R[:, j], j]
        #calc new correlation and corr error
        Cic = np.correcoef(Yi.T)
        E0 = np.copy(E1)
        E1 = sum_Cerr(C=Cic, Ctar=S) # Ctar? or separate variable?
        Serr_iter = np.append(Serr_iter, E1)
        # calculate new Chol and Chol error
        Hic = cholesky(Cic)
        err = Hta[ndx] - Hic[ndx] # Original target - reordered
        Hpr[ndx] = Hpr[ndx] + err

    return Yi, Serr_iter

# def max_err(C, Ctar):
    # """max correlation discrepancy to track"""
    # return np.max(abs(Ctar - C))

# def ILSreorder(Y, Ctar, n_iter=10000):
    # """Iterated Local Search reordering"""
    # n_sim = Y.shape[0]  # Yic is not going to be modified
    # # initalization
    # err_iter = np.empty(shape=(n_iter+1))
    # Emax_iter = np.empty(shape=(n_iter+1))
    # col_iter = np.empty(shape=(n_iter+1))
    # Yi = np.copy(Y)  # usually, preordered simlulations from IC
    # C0 = np.corrcoef(Y.T)
    # E0 = sum_Cerr(C0, Ctar=Ctar)
    # err_iter[0] = E0 # store/track even attemps not accepted
    # Emax_iter[0] = max_err(C0, Ctar=Ctar)
    # Cerr = abs(C0 - Ctar)
    # col_iter[0] = np.argmax(np.apply_along_axis(np.sum, 0, Cerr))
    # # iterative optimziation
    # for ii in range(1, n_iter+1):
        # # identify problematic column with largest disrepencies
        # Cerr = abs(C0 - Ctar)
        # ndx_max = np.argmax(np.apply_along_axis(np.sum, 0, Cerr))
        # col_iter[ii] = ndx_max
        # # randomly select two components of xj to exhcnage and guarantee that we do not eveluate the same neighbor twice
        # rnd_2ndx = rng.choise(n_sim, size=2, replace=False)
        # Ycand = np.copy(Yi)
        # Ycand[rnd_2ndx, ndx_max] = Ycand[np.flip(rnd_2ndx),ndx_max]
        # #recalc actual corr and error
        # C1 = np.corrcoef(Ycand.T)
        # E1 = sum_Cerr(C1, Ctar=Ctar)
        # err_iter[ii] = E1
        # Emax_iter[ii] = max_err(C1, Ctar=Ctar)
        # # save reordered sims if error decreases
        # if E1 <E0:
            # Yi = np.copy(Ycand)
            # E0 = E1
            # C0 = C1

    # return Y1, err_iter, C0, Emax_iter, col_iter



if __name__ == "__main__":

    X = np.array([
        np.linspace(0., 99., 100),
        np.linspace(0., 99., 100),
        np.linspace(0., 99., 100),
        np.linspace(0., 99., 100)
    ]).T

    Ctar = np.array(
        [
            [ 1.00, 0.50, 0.25, 0.05],
            [ 0.50, 1.00, 0.00, 0.30],
            [ 0.25, 0.00, 1.00, 0.00],
            [ 0.05, 0.30, 0.00, 1.00]
        ]
    )

    IC_IPCreorder(X, Ctar)

    # # Iman Conover (fast)
    # # Iterated Perturbed Cholesky (int)
    # # Iterated Local Search (slow)
    # Yic = ICreorder(X=X, Ctar=S)
    # Cic = np.corrcoef(Yic.T)
    # print(f"Iman Conover correlation disrepency {np.round(np.max(abs(Cic - S)), 3)}")

    # Ypc, Epc = IC_IPCreorder(X=X, Ctar=S)  # includes IC as first step
    # Cpc = np.corrcoef(Ypc.T)
    # print(f"Iterated Perturbed Cholesky correlation disrepency {np.round(np.max(abs(Cpc - S)), 3)}")

    # Yls, Els, Cls, Mls, col_err = ILSreorder(Ypc, Ctar=S, n_iter = n_sims * 2) . # start at IPC solution
    # print(f"Iterated Local Search Cholesky correlation disrepency {np.round(np.max(abs(Cls - S)), 3)}")

