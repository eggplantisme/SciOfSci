import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, diags, issparse, csr_array


class SpectralOperator(object):
    def __init__(self):
        self.operator = None
        self.evals = None
        self.evecs = None

    def find_k_eigenvectors(self, K, which='SA', return_eigenvectors=True):
        M = self.operator
        if M.shape[0] == 1:
            print('WARNING: Matrix is a single element')
            evals = np.array([1])
            evecs = np.array([[1]])
        elif K < M.shape[0]:
            if return_eigenvectors is True:
                evals, evecs = eigsh(M, K, which=which, tol=1e-6, return_eigenvectors=return_eigenvectors)
            else:
                evals = eigsh(M, K, which=which, tol=1e-6, return_eigenvectors=return_eigenvectors)
        else:
            if return_eigenvectors is True:
                evals, evecs = eigsh(M, M.shape[0] - 1, which=which, return_eigenvectors=return_eigenvectors)
            else:
                evals = eigsh(M, K, which=which, tol=1e-6, return_eigenvectors=return_eigenvectors)
        self.evals = evals
        if return_eigenvectors is True:
            self.evecs = evecs 

    def find_negative_eigenvectors(self, K_max=None):
        """
        Find negative eigenvectors.

        Given a matrix M, find all the eigenvectors associated to negative
        eigenvalues and return number of negative eigenvalues
        """
        M = self.operator
        Kmax = M.shape[0] - 1 if K_max is None else K_max
        K = min(10, Kmax)
        if self.evals is None:
            self.find_k_eigenvectors(K, which='SA')
        elif len(self.evals) < K:
            self.find_k_eigenvectors(K, which='SA')
        relevant_ev = np.nonzero(self.evals < 0)[0]
        while relevant_ev.size == K and K != Kmax:
            K = min(10 * K, Kmax)  # adjust search speed
            print(f"Try first {K} eigenvalue...")
            self.find_k_eigenvectors(K, which='SA')
            relevant_ev = np.nonzero(self.evals < 0)[0]
            # search negative eigenvectors from 10, 20, 40, ..., use this way to save the memory @jiaze
        self.evals = self.evals[relevant_ev]
        self.evecs = self.evecs[:, relevant_ev]
        return len(relevant_ev)


class BetheHessian(SpectralOperator):

    def __init__(self, A, r=None, regularizer='BHa'):
        super().__init__()
        self.A = A
        self.r = r
        self.calc_r(regularizer)
        self.build_operator()

    def calc_r(self, regularizer='BHa'):
        if self.r is None:
            A = self.A
            if regularizer.startswith('BHa'):
                # set r to square root of average degree
                self.r = np.sqrt(A.sum() / A.shape[0])
        # if last character is 'n' then use the negative version of the BetheHessian
        if regularizer[-1] == 'n':
            self.r = -self.r

    def build_operator(self):
        """
        Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
        B = (r^2-1)*I-r*A+D
        """
        A = self.A
        r = self.r
        A = test_sparse_and_transform(A)
        d = A.sum(axis=1).flatten().astype(float)
        B = eye(A.shape[0]) * (r ** 2 - 1) - r * A + diags(d, 0)
        self.operator = B


class WeightedBetheHessian(BetheHessian):
    def __init__(self, A, r=None, regularizer='BHa'):
        super().__init__(A, r, regularizer)

    def build_operator(self):
        """
        Construct Weighted Bethe Hessian, e.g., in Saade et al
        B_ij = \delta_ij(1 + \sum_{k\in\partial i}\frac{w_ik^2}{r^2-w_ik^2})-\frac{rw_ijA_ij}{r^2-w_ij^2}
        """
        n = self.A.shape[0]
        A = self.A / self.A.max()  # Normalize
        A = self.r * self.A.tanh()
        # print("Weighted BH building...")
        d = csr_array(A ** 2 / (csr_array(self.r ** 2 * np.ones((n, n))) - A ** 2)).sum(axis=1).flatten().astype(float)
        d = diags(d, 0)
        d = d + csr_array(np.identity(n))
        B = d - csr_array((self.r * A) / (csr_array(self.r ** 2 * np.ones((n, n))) - A ** 2))
        print(f"r={self.r}, Weighted BH build.")
        self.operator = B


def test_sparse_and_transform(A):
    """ Check if matrix is sparse and if not, return it as sparse matrix"""
    if not issparse(A):
        print("""Input matrix not in sparse format, transforming to sparse matrix""")
        A = csr_array(A)
    return A
