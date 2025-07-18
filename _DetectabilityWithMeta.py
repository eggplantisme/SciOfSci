import numpy as np
import random
import itertools
import networkx as nx
from numpy.linalg import eig


class MetaSBM:
    def __init__(self, n, rho, ps=None, sizes=None, quickwithoutmeta=False):
        """
        A stochastic block model with metadata generated based on groups
        :param n: number of nodes
        :param rho: alignment probability between meta and group
        :param ps: edge probability between different groups
        :param sizes: number of nodes in each group of each meta, a list of lists, for example:
                      [[10, 10, 10], [10, 10, 10], [10, 10 ,10]] means 3 metadata, each metadata has 3 group,
                      each group 10 nodes.
        """
        self.N = n
        self.rho = rho
        self.sizes = None
        self.ps = None
        self.get_sbm_parameter(sizes, ps)
        self.groupId = None
        self.metaId = None
        if not quickwithoutmeta:
            self.generate_meta()

    def get_sbm_parameter(self, sizes, ps):
        if sizes is None and ps is None:
            # default sizes [[N/4, N/4], [N/4, N/4]]
            number_of_Meta = 2
            number_of_GroupPerMeta = 2
            number_of_group = number_of_Meta * number_of_GroupPerMeta
            if self.N < number_of_group:
                print("Number of Node Too less!")
            else:
                number_of_NodesPerGroup = int(self.N / number_of_group)
                self.sizes = [[number_of_NodesPerGroup] * number_of_GroupPerMeta
                              for i in range(number_of_Meta)]
            # default ps pin=0.8, pout=0.2
            pin = 0.8
            pout = 0.2
            self.ps = (pin - pout) * np.identity(number_of_group) + \
                       pout * np.ones((number_of_group, number_of_group))
        elif sizes is None or ps is None:
            print("Please define both sizes and ps!")
        else:
            self.sizes = sizes
            self.ps = ps

    def generate_meta(self):
        number_of_group = np.sum([len(s) for s in self.sizes])
        number_of_meta = len(self.sizes)
        self.groupId = []
        gid = 0
        for s in self.sizes:
            for size in s:
                self.groupId += [gid]*int(size)
                gid += 1
        self.groupId = np.array(self.groupId)
        # self.groupId = np.arange(0, number_of_group).repeat(np.array([gs for ms in self.sizes for gs in ms]))
        # self.metaId = np.arange(0, number_of_meta).repeat(np.array([np.sum(ms) for ms in self.sizes]))
        self.metaId = []
        mid = 0
        for s in self.sizes:
            self.metaId += [mid]*int(np.sum(s))
            mid += 1
        self.metaId = np.array(self.metaId)
        # print(self.sizes)
        metaIdClass = np.unique(self.metaId)
        for i in range(self.N):
            r = random.random()
            if r > self.rho:
                self.metaId[i] = np.random.choice(np.setdiff1d(metaIdClass, np.array([self.metaId[i]])))
        print("Metadata generation done!")

    def sample(self):
        # group_sizes = np.array(self.sizes).flatten()
        group_sizes = list(itertools.chain.from_iterable(self.sizes))
        g = nx.stochastic_block_model(sizes=group_sizes, p=self.ps)
        A = nx.to_scipy_sparse_array(g)  # csr sparse matrix
        return A

    def filter(self, A, metaId):
        """
        filter the network with specific metaId
        :param A: adjacent matrix by sample
        :param metaId: specific metaId
        :return: filtered matrix and corresponding groupId
        """
        metaIndex = np.where(self.metaId == metaId)[0]
        filterA = A[np.ix_(metaIndex, metaIndex)]
        filterGroupId = self.groupId[metaIndex]
        return filterA, filterGroupId

    @staticmethod
    def get_lambdas(n, rho, Z_s, Z_b, pin, pout):
        """
        PQ Eigenvalues of subgraph in Minority Paper
        :param n: Nodes number in subgraph
        :param rho:
        :param Z_s:
        :param Z_b:
        :param pin:
        :param pout:
        :return:
        """
        delta = pin - pout
        epsilon = rho / Z_s - (1-rho) / Z_b
        lambda1 = (n / 2) * ((rho / Z_s + (1 - rho) / Z_b) * delta + pout +
                             np.sqrt((epsilon * delta + pout) ** 2 + 4 * (rho - 1) * epsilon * delta * pout))
        lambda2 = n * rho / Z_s * delta
        lambda3 = (n / 2) * ((rho / Z_s + (1 - rho) / Z_b) * delta + pout -
                             np.sqrt((epsilon * delta + pout) ** 2 + 4 * (rho - 1) * epsilon * delta * pout))
        lambda4 = n * (1 - rho) / Z_b * delta
        if rho == 0:
            result = [lambda1]
            result.append(lambda4) if Z_b != 1 else None
        elif rho == 1:
            result = [lambda1]
            result.append(lambda2) if Z_s != 1 else None
        elif rho >= Z_s / (Z_s + Z_b):
            result = [lambda1]
            result.append(lambda2) if Z_s != 1 else None
            result.append(lambda3)
            result.append(lambda4) if Z_b != 1 else None
        else:
            result = [lambda1]
            result.append(lambda4) if Z_b != 1 else None
            result.append(lambda3)
            result.append(lambda2) if Z_s != 1 else None
        return result

    @staticmethod
    def get_lambdas_homoDegree(n, d, Z_s, Z_b, rho, delta, Type=2):
        if Type == 2:
            lambda_1 = d
            lambda_2 = n * (1 - rho) / Z_b * delta
            lambda_3 = n * (rho * (1 - rho) / (1 - 2 * rho)) * (1 / Z_s - 1 / Z_b) * delta
            lambda_4 = n * rho / Z_s * delta
            eigva = [lambda_1, lambda_2, lambda_3, lambda_4]
            sorteigva = sorted(eigva, key=lambda v: np.abs(v), reverse=True)
            return sorteigva

    @staticmethod
    def get_lambdas_general(P, Q):
        eigva, _ = eig(P.dot(Q))
        sorteigva = sorted(eigva, key=lambda v: np.abs(v), reverse=True)
        sorteigva = [eigv.real if isinstance(eigv, complex) and abs(eigv.imag) < 1e-4 else eigv for eigv in sorteigva]
        return sorteigva

    def general_get_snr(self, withMeta=False):
        """ Get SNR with general method
        withMeta: get the SNR from subgraph with meta=0
        """
        if withMeta is False:
            P = np.diag(np.array(self.sizes).flatten()) / self.N
            Q = self.N * self.ps
            snr = self.general_SNR(P, Q)
        else:
            X = len(self.sizes)
            Z = len(self.sizes[0])
            if self.rho == 0:
                P = np.diag([(1 - self.rho) / (Z * (X - 1))] * (Z * (X - 1)))
                Q = self.N / X * self.ps[Z, :][Z, :]
            elif self.rho == 1:
                P = np.diag([self.rho / Z] * Z)
                Q = self.N / X * self.ps[:, Z][:, Z]
            else:
                P = np.diag([self.rho / Z] * Z + [(1 - self.rho) / (Z * (X - 1))] * (Z * (X - 1)))
                Q = self.N / X * self.ps
            snr = self.general_SNR(P, Q)
        return snr

    @staticmethod
    def general_SNR(P, Q):
        eigva, _ = eig(P.dot(Q))
        sorteigva = sorted(eigva, key=lambda v: np.abs(v))
        snr = (sorteigva[-2] ** 2) / sorteigva[-1]
        return snr


class SymMetaSBM(MetaSBM):
    def __init__(self, n, X, Z, rho, pin, pout, quickwithoutmeta=False):
        sizes = [[int(n/(X*Z)) for z in range(Z)] for x in range(X)]
        ps = (pin - pout) * np.identity(X*Z) + pout * np.ones((X*Z, X*Z))
        self.pin = pin
        self.pout = pout
        self.X = X
        self.Z = Z
        self.delta = pin - pout
        self.d = n / (X * Z) * (pin + (X * Z - 1) * pout)
        super().__init__(n, rho, ps, sizes, quickwithoutmeta)

    def get_snr(self, withMeta=False, withTheory=True):
        if not withTheory:
            snr = super().get_snr(withMeta)
        else:
            if withMeta is False:
                snr = self.N**2 * self.delta**2 / ((self.X * self.Z)**2*self.d)
            elif self.X == 2:
                # only know for X=2
                temp_sqrtvar = np.sqrt((2*self.Z*self.d-self.N*self.delta)**2+(8*self.N*self.Z*self.d*self.delta)*((1-2*self.rho)**2))
                if self.Z == 1:
                    snr = (2*self.Z*self.d+self.N*self.delta - temp_sqrtvar) ** 2 / (8 * (2*self.Z*self.d+self.N*self.delta + temp_sqrtvar))
                elif self.rho <= 1 / 2:
                    # SNR1
                    snr = 2*(self.N**2)*((1-self.rho)**2)*(self.delta**2)/(self.Z*(2*self.Z*self.d+self.N*self.delta+temp_sqrtvar))
                else:
                    # SNR2
                    snr = 2*(self.N**2)*(self.rho**2)*(self.delta**2)/(self.Z*(2*self.Z*self.d+self.N*self.delta+temp_sqrtvar))
            else:
                print("Theory SNR_SUB for X>2 is not defined!")
                snr = None
        return snr


def test_main():
    X = 2
    Z = 1
    n = X * Z * 2000  # 12000 nodes
    d = 50
    rho = 0.02
    delta = 0.025
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    print(f'SNR_FULL is {msbm.get_snr(False)}, SNR_SUB is {msbm.get_snr(True)}')


if __name__ == '__main__':
    test_main()
