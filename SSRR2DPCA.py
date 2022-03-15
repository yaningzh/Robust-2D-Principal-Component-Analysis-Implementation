"""Structured sparsity regularized robust 2D principal component analysis."""

# Authors: Shion Matsumoto   <matsumos@umich.edu>
#          Rohan Sinha Varma <rsvarma@umich.edu>
#          Marcus König      <marcusko@umich.edu>
#          Yaning Zhang      <yaningzh@umich.edu>

import pdb
import numpy as np
import spams
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def iterBDD(X, E, U, V, r, c, tol=1e-12, max_iter=200, array_fname=None):
    """Iterative bi-directional decomposition.

    Step 1 of the two-stage alternating minimization algorithm for the
    solution to the optimization problem of structured sparsity regularized
    robust 2D-PCA.

    Parameters
    ----------
    X : array, shape (n_samples, m, n)
        Data

    E : array, shape (n_samples, m, n)
        Structured sparse matrices. If none provided, set to E = 0_(m, n) as stated by Sun et al. (2015)

    U : array, shape(U)
        Initial left projection matrix. If none provided, set to U = [ I_(r,r) | 0_(r,m-r) ] as stated
        by Sun et al. (2015)

    V : array, shape(V)
        Initial right projection matrix. If none provided, set to V = [ I_(c,c) | 0_(c,n-c) ] as stated
        by Sun et al. (2015)

    r : int
        Row dimensionality reduction

    c : int
        Column dimensionality reduction

    tol : float
        Tolerance criterion for convergence

    max_iter : int
        Maximum number of iterations

    Returns
    -------
    U : array, shape (m, r)
        Final left projection matrix

    V : array, shape (n, c)
        Final right projection matrix
    """
    pbar = tqdm(range(max_iter))

    # Dimensions of data
    n_samples, m, n = X.shape

    # Calculate (Xi - Ei)
    XE = X - E

    # Update U and V iteratively
    for ii in pbar:

        # Save previous estimates to calculate change
        U_old = U
        V_old = V

        # pdb.set_trace()
        # Eigendecomposition of Cv
        Cv = np.mean([x @ x.T for x in XE.dot(V)], axis=0)
        _, eigvec_u = eigh(Cv)
        eigvec_u = np.fliplr(eigvec_u)

        # Eigendecomposition of Cu
        XET = XE.reshape((n_samples, n, m))
        Cu = np.mean([x @ x.T for x in XET.dot(U)], axis=0)
        _, eigvec_v = eigh(Cu)
        eigvec_v = np.fliplr(eigvec_v)

        # Update U and V
        U = eigvec_u[
            :,
            :r,
        ]
        V = eigvec_v[:, :c]

        # Check convergence
        res_U = calc_residual(U, U_old, relative=False)
        res_V = calc_residual(V, V_old, relative=False)
        pbar.set_postfix({"Residuals (U,V):": [res_U, res_V]})

        if res_U < tol and res_V < tol:
            print("Converged at iteration {}".format(ii + 1))

            # Option to save U and V
            if array_fname is not None:
                print("Saving arrays U and V...")
                np.savez(array_fname, U=U, V=V)

            return U, V

    print("Finished {} iterations. Did not converge.".format(max_iter))
    print("")

    # Option to save U and V
    if array_fname is not None:
        print("Saving arrays U and V...")
        np.savez(array_fname, U=U, V=V)

    return U, V


def feature_outlier_extractor(X, U, V, E, tol=1.0, max_iter=20, array_fname=None):
    """Feature matrix and structured outlier extraction.

    Step 2 of the two-stage alternating minimization algorithm for the
    solution to the optimization problem of structured sparsity
    regularized robust 2D-PCA.

    Parameters
    ----------
    X : array, shape (n_samples, m, n)
        Data

    E : array, shape (n_samples, m, n)
        Structured sparse matrices

    U : array, shape (m, r)
        Initial left projection matrix

    V : array, shape (n, c)
        Initial right projection matrix

    tol : float
        Tolerance criterion for convergence

    max_iter : int
        Maximum number of iterations

    Returns
    -------
    S : array, shape (n_samples, m, n)
        Feature matrix

    E : array, shape (n_samples, m, n)
        Structured sparse outliers matrix
    """
    n_samples, m, n = X.shape
    pbar = tqdm(range(max_iter))

    # Set parameters for proximalFlat method
    lambda2 = 1 / np.sqrt(m * n)  # l1-norm parameter
    lambda1 = lambda2  # sparsity-inducing norm group parameter
    groups = create_groups(m, n, ravel=True)
    param = {
        "lambda1": lambda1,
        "lambda2": lambda2,
        # "groups": groups,
        "size_group": 3,
        "regul": "sparse-group-lasso-l2",
    }

    for ii in pbar:

        # Save previous estimates to calculate change
        E_old = E

        # Bi-directional projection
        S = np.array([U.T @ x @ V for x in X - E])

        # Calculate input signal and reshape each sample into a column vector
        YUSVT = np.array([x - U @ s @ V.T for x, s in zip(X, S)])
        YUSVT = YUSVT.reshape((n_samples, m * n))

        # pdb.set_trace()
        # Proximal gradient method to solve structured sparsity regularized problem
        e = spams.proximalFlat(np.asfortranarray(YUSVT), **param)

        # Reshape sparse outliers term
        E = e.reshape((n_samples, m, n))

        # Calculate residuals
        res_S = 0
        res_E = calc_residual(E, E_old)
        pbar.set_postfix({"Residuals (S,E):": [res_S, res_E]})

        # Check convergence
        if res_S < tol and res_E < tol:
            print("Converged at iteration {}".format(ii + 1))

            # Option to save S and E
            if array_fname is not None:
                print("Saving arrays S and E...")
                np.savez(array_fname, S=S, E=E)

            return S, E

    print("Finished {} iterations. Did not converge.".format(max_iter))
    print("")

    # Option to save S and E
    if array_fname is not None:
        print("Saving arrays S and E...")
        np.savez(array_fname, S=S, E=E)

    return S, E


def create_groups(m, n, ravel=True):
    """Create indices for 3-by-3 grids to specify group structure for structured sparsity regularization"""
    indices = np.array(
        [
            [
                [i - 1, j - 1],
                [i - 1, j],
                [i - 1, j + 1],
                [i, j - 1],
                [i, j],
                [i, j + 1],
                [i + 1, j - 1],
                [i + 1, j],
                [i + 1, j + 1],
            ]
            for i in range(1, m - 1)
            for j in range(1, n - 1)
        ]
    )
    if ravel:
        indices_raveled = np.array(
            [np.ravel_multi_index(idx_list.T, (m, n)) for idx_list in indices]
        )
        return indices_raveled
    else:
        return indices


def calc_residual(Y, Y1, relative=True):
    """Calculate residual"""
    if Y.shape == Y1.shape and Y.ndim > 2:
        Y = Y.reshape((Y.shape[0], -1))
        Y1 = Y1.reshape((Y1.shape[0], -1))
    if relative:
        return np.linalg.norm(Y - Y1, ord="fro") / np.linalg.norm(Y, ord="fro")
    else:
        return np.linalg.norm(Y - Y1, ord="fro")


def ssrr2dpca(X, scale, UV_file=None):
    """Structured sparsity regularized robust 2D principal component
    analysis (SSR-R2D-PCA).
    """
    X = X.astype(float)

    # Get dimensions of data (following notations of paper)
    T, m, n = np.shape(X)

    # # Scale values to lie within [0,1]
    # X_scaled = MinMaxScaler().fit_transform(X, )

    # Center data
    X -= X.mean(axis=0)

    # Calculate dimension reduction parameters
    r = int(m / scale)
    c = int(n / scale)

    E = np.tile((np.zeros((1, m, n))), (T, 1, 1))

    # Get left and right projection matrices
    l = 1 / np.sqrt(m * n)
    if UV_file is not None:
        print("Loading U and V from {}...".format(UV_file))
        npzfile = np.load(UV_file)
        U = npzfile["U"]
        V = npzfile["V"]
    else:
        print("Calculating iterative bi-directional decomposition...")
        # Initialize projection and structured sparse matrices
        U = np.vstack((np.eye(r, r), np.zeros((m - r, r))))  # shape(U) = (m,r)
        V = np.vstack((np.eye(c, c), np.zeros((n - c, c))))  # shape(V) = (n,c)
        U, V = iterBDD(X, E, U, V, r, c)

    # Get feature matrix and structured outliers
    print("Performing feature matrix and structured outlier extraction...")
    S, E = feature_outlier_extractor(X, U, V, E)

    return U, V, S, E


def reconstruct(U, V, S, E):
    """Reconstruct using decomposition

    Xi = U @ Si @ V.T

    Parameters
    ----------
    U : array, shape (m, r)

    S : array, shape (n_samples, r, c)

    V : array, shape (n, c)

    Returns
    -------
    recon : array, shape (n_samples, m, n)
    """
    # recon = np.einsum("ij,ljk->lik", U, S.dot(V.T)) + E
    recon = np.array([U.dot(Si.dot(V.T)) for Si in S])
    return recon


class SSRR2DPCA:
    """Structured sparsity regularized robust 2D principal component
    analysis (SSR-R2D-PCA).

    Dimensionality reduction technique based on the algorithm detailed in
    "Robust 2D principal component analysis: A structured sparsity regularized
    approach" by Yipeng Sun, Xiaoming Tao, Yang Li, and Jianhua Lu.

    The construction of this class is based on scikit-learn's PCA class.

    Parameters
    ----------
    n_components : int
        Number of principal components

    l : float
        Sparsity regularization term

    b : array, shape (n_components)
        Structured sparsity regularization term, default value of l when w=1

    U : array, shape (nx, ny, n_components)
        Left projection matrices

    V : array, shape (nx, ny, n_components)
        Right projection matrices

    E : array, shape (nx, ny, n_samples)
        Structured outliers

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal components

    explained_variance_ : array, shape (n_components,)
        Amount of variance explained by each of the principal components

    explained_variance_ratio_ : array, shape (n_components,)
        Ratio (percentage) of variance explained by each of the principal
        components

    singular_values_ : array, shape (n_components,)
        Singular values associated with each principal component

    mean_ : array, shape (nx, ny)
        Empiricial mean estimated from training data

    n_components_ : int
        Number of principal components

    n_feautres_ : array, shape (2,)
        Number of features in x- and y-directions

    n_samples_ : int
        Number of samples in the training data

    References
    ----------
    Yipeng Sun et al. “Robust 2D principal component analysis:
    A structured sparsity regularized approach”.
    In:IEEE Transactions on Image Processing 24.8 (Aug. 2015), pp. 2515–2526.
    ISSN:10577149.DOI:10.1109/TIP.2015.2419075.
    """

    def __init__(self, r=None, c=None, lam=None, beta=None):
        self.r_ = r
        self.c_ = c
        self.lam_ = lam
        self.beta_ = beta
        self.U_ = None
        self.V_ = None
        self.S_ = None
        self.E_ = None

    def fit(self, X):
        """Fit model with data X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features_x, n_features_y)
            Training data used to fit model

        Returns
        -------
        self : object
            Returns object instance
        """
        X = X.astype(float)

        # Get dimensions of data (following notations of paper)
        T, m, n = np.shape(X)
        r, c = self.n_components_x_, self.n_components_y_
        self.n_samples_, self.n_features_y_, self.n_features_x_ = T, m, n

        # # Scale values to lie within [0,1]
        # X_scaled = MinMaxScaler().fit_transform(X, )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Initialize projection and structured sparse matrices
        U = np.vstack((np.eye(r, r), np.zeros((m - r, r))))  # shape(U) = (m,r)
        V = np.vstack((np.eye(c, c), np.zeros((n - c, c))))  # shape(V) = (n,c)
        E = np.tile((np.zeros((1, m, n))), (T, 1, 1))

        # Get left and right projection matrices
        self.U, self.V = iterBDD(X, E, U, V)

        # Get feature matrix and structured outliers
        self.S, self.E = feature_outlier_extractor(X, self.U, self.V, tol=1e-3)

        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data

        Returns
        -------
        X_transformed : shape (n_samples, n_components)
            X transformed
        """
        return 0

    def fit_transform(self, X):
        """Fit model with data X and apply dimensionality reduction to X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data used to fit model

        Returns
        -------
        X_transformed : array (n_samples, n_components)
            X transformed using dimensionality reduction
        """
        self.fit(X)
        self.transform(X)

        return 0
