import numpy as np
from scipy import sparse
from scipy.linalg import fractional_matrix_power
from scipy.sparse import eye, diags, issparse, identity


class Propagation:
    def __init__(self, A, dim=1):
        """
        Base class for propagation process
        :param A: a csr-sparse matrix
        :param dim:
        """
        self.A = A
        self.n = A.shape[0]
        self.operator = None
        self.build_operator()
        self.signal = None
        self.init_signal(dim)

    def build_operator(self):
        self.operator = self.A

    def init_signal(self, dim):
        self.signal = np.random.rand(self.n, dim)

    def normalize(self):
        """
        Normalize self.signal by row
        :return:
        """
        Z_inverse = np.diag(1 / np.sum(self.signal, 1))
        self.signal = np.dot(Z_inverse, self.signal)

    def propagate(self):
        self.signal = self.operator.dot(self.signal)


class TwoStepLabelPropagation(Propagation):
    def __init__(self, A, dim, B, alpha=0.5, beta=1, operator_name="L^2"):
        """
        See paper “Graph-based semi-supervised learning for relational networks”
        """
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.operator_name = operator_name
        super().__init__(A, dim)
        self.normalize()

    def build_operator(self):
        d = self.A.sum(axis=1).flatten().astype(float)
        if self.operator_name == "L^2":
            # operator = diag(1/\sqrt{d}) A diag(1/\sqrt{d})
            d_sqrt_inverse = diags(1/np.sqrt(d))
            L = d_sqrt_inverse.dot(self.A)
            L = L.dot(d_sqrt_inverse)
            self.operator = np.linalg.matrix_power(L.dot(L).toarray(), self.beta)
        elif self.operator_name == "W^2":
            dmax = np.max(d)
            L = diags(d) - self.A
            W = identity(self.n) - 1 / dmax * L
            self.operator = W.dot(W).toarray()

    def propagate(self):
        self.signal = self.alpha * self.operator.dot(self.signal) + (1-self.alpha) * self.B
        self.normalize()

    def result(self):
        return np.argmax(self.signal, axis=1)

