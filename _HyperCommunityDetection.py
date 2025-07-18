import numpy as np
import time
from _HyperSBM import UniformSymmetricHSBM
from spectralOperator import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _CommunityDetect import CommunityDetect
# from EXPERIMENT_HYPER_EMPIRICAL import EmpiricalHyperGraph
from other.hypergraph_message_passing.jiaze_main_message_passing import *


class HyperCommunityDetect:
    def __init__(self):
        pass

    @staticmethod
    def BetheHessian(hsbm, num_groups=None, only_assortative=False, consider_ks=None, projectionMatrix=None, givenBulk=None, BHposOperator=None):
        start = time.time()
        d = hsbm.H.sum() / hsbm.n
        if givenBulk is None:
            if type(hsbm) is UniformSymmetricHSBM:
                bulk = np.sqrt(d * (hsbm.k - 1))
            else:
                # For Nonuniform Hsbm or empirical hyper graph
                edge_order, edge_count = np.unique(hsbm.H.sum(axis=0).flatten(), return_counts=True)
                order_count = dict(zip(edge_order, edge_count))
                # print(order_count)
                ds = dict()
                for o in order_count:
                    ds[o] = o * order_count[o] / hsbm.n
                bulk = 0
                for k in hsbm.Ks:
                    bulk += ds[k] * (k - 1)
                bulk = np.sqrt(bulk)
        else:
            bulk = givenBulk
        BHa_pos = SpectralOperator()
        if projectionMatrix is not None:
            BHa_pos.operator = HyperSBM.get_projection_operator(projection_matrix=projectionMatrix, operator='WBH',
                                                                r=bulk)
        elif "EmpiricalHyperGraph" in str(type(hsbm)):
            BHa_pos.operator = hsbm.get_operator("BH", r=bulk, consider_ks=consider_ks)
        else:
            if BHposOperator is None:
                BHa_pos.operator = hsbm.get_operator("BH", r=bulk, consider_ks=consider_ks)
            else:
                BHa_pos.operator = BHposOperator
        if only_assortative is False:
            BHa_neg = SpectralOperator()
            if projectionMatrix is not None:
                BHa_neg.operator = HyperSBM.get_projection_operator(projection_matrix=projectionMatrix, operator='WBH',
                                                                    r=-bulk)
            elif "EmpiricalHyperGraph" in str(type(hsbm)):
                BHa_neg.operator = hsbm.get_operator("BH", r=-bulk, consider_ks=consider_ks)
            else:
                BHa_neg.operator = hsbm.get_operator("BH", r=-bulk)
        if num_groups is None:
            Kpos = BHa_pos.find_negative_eigenvectors()
            if only_assortative is False:
                Kneg = BHa_neg.find_negative_eigenvectors()
                num_groups = Kpos + Kneg if Kpos + Kneg < hsbm.n else hsbm.n  # max number of group should be N
                print(f'number of groups = {num_groups}, Kpos={Kpos}, Kneg={Kneg}')
            else:
                num_groups = Kpos
                print(f'number of groups = {num_groups}, Kpos={Kpos}')
            if num_groups == 0 or num_groups == 1:
                print("no indication for grouping -- return all in one partition")
                partition_vecs = np.zeros(hsbm.n, dtype='int')
                return partition_vecs, num_groups
            # construct combined_evecs to cluster
            if only_assortative is False:
                combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
            else:
                combined_evecs = BHa_pos.evecs
        else:
            # If num_group is given, cluster evec corresonding with the first num_group eval of BHa_pos and BHa_neg
            BHa_pos.find_k_eigenvectors(num_groups, which='SA')
            if only_assortative is False:
                BHa_neg.find_k_eigenvectors(num_groups, which='SA')
            # combine both sets of eigenvales and eigenvectors and take first k
            if only_assortative is False:
                combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
                combined_evals = np.hstack([BHa_pos.evals, BHa_neg.evals])
            else:
                combined_evecs = BHa_pos.evecs
                combined_evals = BHa_pos.evals
            index = np.argsort(combined_evals)
            combined_evecs = combined_evecs[:, index[:num_groups]]
        print(f"EVECs construct: {time.time() - start}")
        # cluster with Kmeans
        if num_groups < hsbm.n:
            cluster = KMeans(n_clusters=num_groups, n_init=20)
            cluster.fit(combined_evecs)
            partition_vecs = cluster.predict(combined_evecs)
        else:
            partition_vecs = np.array(list(range(hsbm.n)))
        return partition_vecs, num_groups

    @staticmethod
    def ProjBH(hsbm, num_groups=None, only_assortative=False):
        projA = hsbm.H.dot(hsbm.H.T)
        projA = projA - diags(projA.diagonal())
        print(f"Construct projected weighted matrix, do weighted BH on it...")
        return CommunityDetect(projA).BetheHessian(num_groups, weighted=True, only_assortative=only_assortative)

    @staticmethod
    def NonBackTracking_(hsbm, num_groups=None, sign=True):
        k_ = len(hsbm.Ks)
        n = hsbm.n
        NB_ = SpectralOperator()
        NB_.operator = hsbm.get_operator("NB_")
        NB_.find_k_eigenvectors(K=num_groups, which="LA")
        X = np.zeros((n, num_groups-1))
        for q in range(1, num_groups):
            evec = NB_.evecs[:, q]
            x = np.zeros(n)
            # construct x
            for i in range(n):
                for ki in range(k_):
                    x[i] += evec[ki * n + i]
            # sign function
            if sign is True:
                for i in range(n):
                    if x[i] > 0:
                        x[i] = 1
                    elif x[i] < 0:
                        x[i] = -1
                    else:
                        x[i] = 0
            X[:, q-1] = x
        cluster = KMeans(n_clusters=num_groups, n_init=20)
        cluster.fit(X)
        partition_vecs = cluster.predict(X)
        return partition_vecs, num_groups

    @staticmethod
    def DCBetheHessian(hsbm, num_groups=None, only_assortative=True, borrowNB=True):
        if borrowNB:
            # get first num_groups lambda(eigenvalue of NB) by NB
            NB = hsbm.get_operator('NB')
            print(f'shape of NB: {np.shape(NB)}')
            if only_assortative:
                which = 'LA'
            else:
                which = 'LM'
            eig_NB, _ = eigsh(NB, num_groups, which=which, tol=1e-6)
            eig_NB = np.sort(eig_NB)[::-1]
            print(f'eigenvalue of NB {eig_NB}')
        combined_evecs = None
        for i, lam in enumerate(eig_NB):
            BH = SpectralOperator()
            BH.operator = hsbm.get_operator("BH", r=lam)
            BH.find_k_eigenvectors(i+1, which='SA')
            if combined_evecs is None:
                combined_evecs = BH.evecs
            else:
                combined_evecs = np.hstack([combined_evecs, BH.evecs[:, i][:, np.newaxis]])
        # cluster with Kmeans
        if num_groups < hsbm.n:
            cluster = KMeans(n_clusters=num_groups, n_init=20)
            cluster.fit(combined_evecs)
            partition_vecs = cluster.predict(combined_evecs)
        else:
            partition_vecs = np.array(list(range(hsbm.n)))
        return partition_vecs, num_groups

    @staticmethod
    def BeliefPropagation(hsbm, args):
        # path = "./other/hypergraph_message_passing/data/jiaze_synthetic/"
        # hsbm.save_txt(path + "test_sample.txt")
        # hsbm.save_parameter(path + "test_parameter.npz")
        hsbm.save_txt(args["hypergraph"])
        hsbm.save_parameter(args["hsbm_parameter"])
        arg = Arguments()
        arg.hye_file = args["hypergraph"]
        arg.n = args["hsbm_parameter"]
        arg.p = args["hsbm_parameter"]
        arg.K = args["q"]
        arg.hye_sizes = args["hyperedge_sizes"]
        arg.save_dir = Path(args["save_dir"])
        arg.dropout = 0
        main0(arg)
        result = np.load(args["save_dir"]+"/inferred_params.npz")
        partition_vec = np.argmax(result["log_marginals"], axis=1)
        return partition_vec, args["q"]


def main_observe_with_debug(result_path):
    result = np.load(result_path)
    pass


def main_bp_debug():
    arg = Arguments()
    arg.hye_file = "./other/hypergraph_message_passing/data/jiaze_synthetic/" \
                   "amiexp_n=100_q=2_d=10_Ks=(2, 3)_epsilon=0.4_times=8_hgraph.txt"
    arg.n = "./other/hypergraph_message_passing/data/jiaze_synthetic/" \
            "amiexp_n=100_q=2_d=10_Ks=(2, 3)_epsilon=0.4_times=8_parameter.npz"
    arg.p = "./other/hypergraph_message_passing/data/jiaze_synthetic/" \
            "amiexp_n=100_q=2_d=10_Ks=(2, 3)_epsilon=0.4_times=8_parameter.npz"
    arg.K = 2
    arg.hye_sizes = [2, 3]
    arg.save_dir = Path("./other/hypergraph_message_passing/mp_results/jiaze")
    arg.dropout = 0.25
    main0(arg)


if __name__ == '__main__':
    main_bp_debug()
    main_observe_with_debug(
        "./other/hypergraph_message_passing/mp_results/jiaze/"
        "inferred_params.npz")

