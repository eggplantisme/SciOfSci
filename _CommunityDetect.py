import numpy as np
import time
import os
import networkx as nx
# import graph_tool.all as gt
from spectralOperator import BetheHessian, WeightedBetheHessian
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _DetectabilityWithMeta import *
from propagation import TwoStepLabelPropagation
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, diags, issparse, csr_matrix
from multiprocessing import Pool


class CommunityDetect:
    def __init__(self, A):
        """
        do community detection to network represented by adjacency matrix A
        :param A: sparse csr matrix
        """
        self.A = A
        self.g = None
        self.d = None
        self.n = None

    def BetheHessian(self, num_groups=None, help_evec=None, help_num_groups=None, return_evec=False, weighted=False, only_assortative=False):
        start = time.time()
        BHa_pos = BetheHessian(self.A, regularizer='BHa') if weighted is False else \
            WeightedBetheHessian(self.A, regularizer='BHa')
        if only_assortative is False:
            BHa_neg = BetheHessian(self.A, regularizer='BHan') if weighted is False else \
                WeightedBetheHessian(self.A, regularizer='BHan')
        N = np.shape(self.A)[0]
        print(f"Get BH operator, done!")
        if num_groups is None:
            Kpos = BHa_pos.find_negative_eigenvectors()
            if only_assortative is False:
                Kneg = BHa_neg.find_negative_eigenvectors()
                num_groups = Kpos + Kneg if Kpos + Kneg < N else N  # max number of group should be N
                print(f'number of groups = {num_groups}, Kpos={Kpos}, Kneg={Kneg}')
            else:
                num_groups = Kpos if Kpos < N else N
                print(f'number of groups = {num_groups}, Kpos={Kpos}')
            if num_groups == 0 or num_groups == 1:
                if help_evec is not None and help_num_groups is not None:
                    print(f"no indication for grouping -- But cluster with help evec and help number of grous ={help_num_groups}")
                    combined_evecs = help_evec
                    if np.shape(combined_evecs)[1] < 1:
                        partition_vecs = np.zeros(self.A.shape[0], dtype='int')
                    else:
                        # cluster with Kmeans
                        cluster = KMeans(n_clusters=help_num_groups, n_init=20)
                        cluster.fit(combined_evecs)
                        partition_vecs = cluster.predict(combined_evecs)
                        num_groups = help_num_groups
                else:
                    print("no indication for grouping -- return all in one partition")
                    partition_vecs = np.zeros(self.A.shape[0], dtype='int')
                    if return_evec:
                        return np.array([[]]*self.A.shape[0]), partition_vecs, num_groups
                return partition_vecs, num_groups
            # construct combined_evecs to cluster
            combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
        else:
            # If num_group is given, cluster evec corresonding with the first num_group eval of BHa_pos and BHa_neg
            BHa_pos.find_k_eigenvectors(num_groups, which='SA')
            if only_assortative is False:
                BHa_neg.find_k_eigenvectors(num_groups, which='SA')
                # combine both sets of eigenvales and eigenvectors and take first k
                combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
                combined_evals = np.hstack([BHa_pos.evals, BHa_neg.evals])
                index = np.argsort(combined_evals)
                combined_evecs = combined_evecs[:, index[:num_groups]]
            else:
                combined_evecs = BHa_pos.evecs
                combined_evals = BHa_pos.evals
                index = np.argsort(combined_evals)
                combined_evecs = combined_evecs[:, index[:num_groups]]
        if help_evec is not None:
            combined_evecs = np.hstack([help_evec, combined_evecs])
        print(f"EVECs construct: {time.time() - start}")
        # cluster with Kmeans
        if num_groups < N:
            cluster = KMeans(n_clusters=num_groups, n_init=20)
            cluster.fit(combined_evecs)
            partition_vecs = cluster.predict(combined_evecs)
        else:
            partition_vecs = np.array(list(range(N)))
        if return_evec:
            return combined_evecs, partition_vecs, num_groups
        return partition_vecs, num_groups

    def DCBetheHessian(self, num_groups=None, help_evec=None, help_num_groups=None, return_evec=False):
        zetas = []
        BHa_pos = BetheHessian(self.A, regularizer='BHa')
        BHa_neg = BetheHessian(self.A, regularizer='BHan')
        Kpos = BHa_pos.find_negative_eigenvectors()
        Kneg = BHa_neg.find_negative_eigenvectors()
        # Get the number of group need to be detected
        if num_groups is None:
            num_groups = Kpos + Kneg
            if num_groups == 0 or num_groups == 1:
                if help_evec is not None and help_num_groups is not None:
                    print(f"no indication for grouping -- But cluster with help evec and help number of grous ={help_num_groups}")
                    combined_evecs = help_evec
                    if np.shape(combined_evecs)[1] < 1:
                        partition_vecs = np.zeros(self.A.shape[0], dtype='int')
                    else:
                        # cluster with Kmeans
                        cluster = KMeans(n_clusters=help_num_groups, n_init=20)
                        cluster.fit(combined_evecs)
                        partition_vecs = cluster.predict(combined_evecs)
                        num_groups = help_num_groups
                else:
                    print("no indication for grouping -- return all in one partition")
                    partition_vecs = np.zeros(self.A.shape[0], dtype='int')
                zetas.append(0)
                return partition_vecs, num_groups, zetas
        print(f'number of groups = {num_groups}')
        # make sure border
        d = self.A.sum(axis=1).flatten().astype(float)
        rho = np.sum(d * d) / np.sum(d)
        border = np.sqrt(rho)
        print("border", border, "Kpos", Kpos, "Kneg", Kneg)
        # bipartite search eta to get eigenvector need to be combined
        count_evecs = (Kpos - 1 if Kpos > 0 else 0) + (Kneg if Kneg > 0 else 0)
        # combined_evecs = np.zeros((np.shape(self.A)[0], count_evecs))
        zetas.append(count_evecs)
        saved_eigen = dict()
        # For r > 0 case
        zeta_p_pos = np.zeros(Kpos - 1 if Kpos > 0 else 0)
        for p in range(2, Kpos + 1):
            # if num_groups == 3:
            #     print("Debug Here!")
            start_r = 1
            end_r = border
            while start_r < end_r and zeta_p_pos[p-2] == 0:
                # save eigen for r to reduce time
                if start_r not in saved_eigen.keys():
                    BHa_start = BetheHessian(self.A, r=start_r, regularizer='BHa')
                    BHa_start.find_k_eigenvectors(Kpos)
                    saved_eigen[start_r] = (BHa_start.evals, BHa_start.evecs)
                if end_r not in saved_eigen.keys():
                    BHa_end = BetheHessian(self.A, r=end_r, regularizer='BHa')
                    BHa_end.find_k_eigenvectors(Kpos)
                    saved_eigen[end_r] = (BHa_end.evals, BHa_end.evecs)
                if np.around(saved_eigen[start_r][0][p-1], 10) * np.around(saved_eigen[end_r][0][p-1], 10) < 0:
                    # the pth eigenvalue has different sign for BH with start_r and end_r\
                    mid_r = (start_r + end_r) / 2
                    if mid_r not in saved_eigen.keys():
                        # start_t = time.time()
                        BHa_mid = BetheHessian(self.A, r=mid_r, regularizer='BHa')
                        BHa_mid.find_k_eigenvectors(Kpos)
                        saved_eigen[mid_r] = (BHa_mid.evals, BHa_mid.evecs)
                        # print(f'r={mid_r}, eigsh time={time.time()-start_t}')
                    if np.around(saved_eigen[mid_r][0][p-1], 10) == 0:
                        zeta_p_pos[p - 2] = mid_r
                        break
                    elif saved_eigen[mid_r][0][p-1] * saved_eigen[start_r][0][p-1] < 0:
                        end_r = mid_r
                    else:
                        start_r = mid_r
                elif np.around(saved_eigen[start_r][0][p-1], 10) == 0:
                    # When in pin >  pout = 0, (2nd, 3rd, ... num of disconnected component) eigenvalue == 0,
                    # but because of the precision, it < 0 (such as -2.62926e-14), so we need to around the eigen.
                    zeta_p_pos[p-2] = start_r
                    break
                elif np.around(saved_eigen[end_r][0][p-1], 10) == 0:
                    zeta_p_pos[p-2] = end_r
                    break
                else:
                    print(r"In positive case, No sign change for $\zeta$ between" + f"{start_r}~{end_r} for {p}th eigenvalue")
                    break
                if abs(end_r - start_r) < 1e-4:
                    if abs(np.around(saved_eigen[start_r][0][p - 1] - 0, 10)) < abs(np.around(saved_eigen[end_r][0][p - 1] - 0, 10)):
                        zeta_p_pos[p - 2] = start_r
                    else:
                        zeta_p_pos[p - 2] = end_r
            print(f'In positive case, Find {p}th eigenvalue = 0 of BH with ' + r'$\zeta$=' + f'{zeta_p_pos[p-2]}')
        # For r < 0 case
        zeta_p_neg = np.zeros(Kneg if Kneg > 0 else 0)
        for p in range(1, Kneg + 1):
            start_r = -1
            end_r = -border
            while start_r > end_r and zeta_p_neg[p - 1] == 0:
                # save eigen for r to reduce time
                if start_r not in saved_eigen.keys():
                    BHa_start = BetheHessian(self.A, r=start_r, regularizer='BHa')
                    BHa_start.find_k_eigenvectors(Kneg)
                    saved_eigen[start_r] = (BHa_start.evals, BHa_start.evecs)
                if end_r not in saved_eigen.keys():
                    BHa_end = BetheHessian(self.A, r=end_r, regularizer='BHa')
                    BHa_end.find_k_eigenvectors(Kneg)
                    saved_eigen[end_r] = (BHa_end.evals, BHa_end.evecs)
                if np.around(saved_eigen[start_r][0][p - 1], 10) * np.around(saved_eigen[end_r][0][p - 1], 10) < 0:
                    # the pth eigenvalue has different sign for BH with start_r and end_r\
                    mid_r = (start_r + end_r) / 2
                    if mid_r not in saved_eigen.keys():
                        BHa_mid = BetheHessian(self.A, r=mid_r, regularizer='BHa')
                        BHa_mid.find_k_eigenvectors(Kneg)
                        saved_eigen[mid_r] = (BHa_mid.evals, BHa_mid.evecs)
                    if np.around(saved_eigen[mid_r][0][p - 1], 10) == 0:
                        zeta_p_neg[p - 1] = mid_r
                    elif saved_eigen[mid_r][0][p - 1] * saved_eigen[start_r][0][p - 1] < 0:
                        end_r = mid_r
                    else:
                        start_r = mid_r
                elif np.around(saved_eigen[start_r][0][p - 1], 10) == 0:
                    zeta_p_neg[p - 1] = start_r
                    break
                elif np.around(saved_eigen[end_r][0][p - 1], 10) == 0:
                    zeta_p_neg[p - 1] = end_r
                    break
                else:
                    print(r"In negative case, No sign change for $\zeta$ between" + f"{start_r}~{end_r} for {p}th eigenvalue")
                    break
                if abs(end_r - start_r) < 1e-4:
                    if abs(np.around(saved_eigen[start_r][0][p - 1] - 0, 10)) < abs(np.around(saved_eigen[end_r][0][p - 1] - 0, 10)):
                        zeta_p_neg[p - 1] = start_r
                    else:
                        zeta_p_neg[p - 1] = end_r
            print(f'In negative case, Find {p}th eigenvalue = 0 of BH with ' + r'$\zeta$=' + f'{zeta_p_neg[p - 1]}')
        # Combined the evecs
        if np.all(zeta_p_pos == 0) and np.all(zeta_p_neg == 0):
            print("DCBH failed for search all zeta!")
            return None, None, None
        else:
            evecs_pos = np.zeros((np.shape(self.A)[0], Kpos - 1 if Kpos > 0 else 0))
            evecs_neg = np.zeros((np.shape(self.A)[0], Kneg if Kneg > 0 else 0))
            for p in range(2, Kpos + 1):
                if zeta_p_pos[p-2] != 0:
                    zetas.append(zeta_p_pos[p-2])
                    evecs_pos[:, p-2] = saved_eigen[zeta_p_pos[p-2]][1][:, p-1]
            for p in range(1, Kneg + 1):
                if zeta_p_neg[p-1] != 0:
                    zetas.append(zeta_p_neg[p-1])
                    evecs_neg[:, p-1] = saved_eigen[zeta_p_neg[p-1]][1][:, p-1]
        combined_evecs = np.hstack([evecs_pos, evecs_neg])
        if help_evec is not None:
            combined_evecs = np.hstack([help_evec, combined_evecs])
        # cluster with Kmeans
        cluster = KMeans(n_clusters=num_groups, n_init=20)
        cluster.fit(combined_evecs)
        partition_vecs = cluster.predict(combined_evecs)
        if return_evec:
            return combined_evecs, partition_vecs, num_groups
        return partition_vecs, num_groups, zetas

    def TwoStepLabelPropagate(self, B, alpha=0.1, operator_name='L^2'):
        k = np.shape(B)[1]
        propagation = TwoStepLabelPropagation(self.A, k, B, alpha=alpha, operator_name=operator_name)
        iter_i = 0
        while True:
            last_F = np.copy(propagation.signal)
            propagation.propagate()
            F = propagation.signal
            diff = np.sum(np.abs(F - last_F))
            print(f'iter {iter_i}, diff={diff}')
            if diff < 1e-6 or iter_i > 200:
                break
            iter_i += 1
        subPropagationPartition = propagation.result()
        num_group = np.size(np.unique(subPropagationPartition))
        return subPropagationPartition, num_group

    def BP_meta(self, num_groups, na, cab, rho, groupId, metaId, processId=None):
        g = nx.from_scipy_sparse_array(self.A)
        size = g.number_of_nodes()
        '''Drop the weights from a networkx weighted graph.'''
        for node, edges in nx.to_dict_of_dicts(g).items():
            for edge, attrs in edges.items():
                attrs.pop('weight', None)
        for node in g:
            # print(node)
            g.nodes[node].pop('label', None)
            g.nodes[node]['value'] = groupId[node]
            g.nodes[node]['meta'] = metaId[node]
        
        net_gml_path = f'./other/meta_mode_net-main/src/data/temp{str(processId) if processId is not None else ""}.gml'
        target_path = f'./other/meta_mode_net-main/src/data/marginal{str(processId) if processId is not None else ""}.txt'
        nx.write_gml(g, net_gml_path, stringizer=str)
        # Set rho in command from 0.01~(1/3+0.01), avoid issue when rho=1 in fullgraph and rho=0,1 in subgraph
        # rho_in_command = rho / 3 + 1e-2
        num_metas = np.size(np.unique(metaId))
        Z = num_groups / num_metas
        rho_in_command = np.zeros((num_groups, num_metas))
        for i in range(num_groups):
            origin_meta = i // Z
            for j in range(num_metas):
                if j == origin_meta:
                    rho_in_command[i][j] = rho
                else:
                    rho_in_command[i][j] = (1 - rho) / (num_metas - 1)
        command = f'./other/meta_mode_net-main/src/sbm infer -l {net_gml_path} -n{size} -q{num_groups} -p{",".join([str(x) for x in na])} -c{",".join([str(x) for x in cab])} -a{",".join([str(i) for i in np.nditer(rho_in_command)])} -M {target_path} -v-1'
        print(command)
        os.system(command)
        with open(target_path, 'r') as f:
            f.readline()
            f.readline()
            line = f.readline()
            partition = np.array([int(i) for i in line.strip().split(' ')])
        return partition

    def BP(self, num_groups, na, cab, groupId, processId=None, infermode=0, init_epsilon=None, learn_conv_crit=None, learn_max_time=None):
        """
        infermode: 
        0 means infer with true parameter
        1 means learn true parameter with random initialization then infer
        2 means learn true parameter with initialization -P0,d then infer
        3 means learn true parameter with given initialization na cab then infer
        4 means learn q with BP before learn parameter with random initialization then infer
        5 means learn q with BP before learn parameter with initialization -P0,d then infer
        6 means learn q with BP before learn parameter with given initialization(epsilon-c mode) then infer
        """
        if self.g is None and self.d is None and self.n is None:
            g = nx.from_scipy_sparse_array(self.A)
            size = g.number_of_nodes()
            '''Drop the weights from a networkx weighted graph.'''
            for node, edges in nx.to_dict_of_dicts(g).items():
                for edge, attrs in edges.items():
                    attrs.pop('weight', None)
                g.nodes[node].pop('label', None)
                g.nodes[node]['value'] = str(groupId[node])
            d = self.A.sum() / self.A.shape[0]
        else:
            g = self.g
            d = self.d
            size = self.n
        
        net_gml_path = f'./other/mode_net/data/temp{str(processId) if processId is not None else ""}.gml'
        target_path = f'./other/mode_net/data/marginal{str(processId) if processId is not None else ""}.txt'
        with open(net_gml_path, 'w') as gw:
            for line in nx.generate_gml(g):
                if 'label' in line:
                    pass
                elif 'value' in line:
                    line = line.replace('\"', '')
                    gw.write(line + '\n')
                else:
                    gw.write(line + '\n')
        # nx.write_gml(g, net_gml_path, stringizer=None)
        if infermode == 0:
            command = f'./other/mode_net/sbm infer -l {net_gml_path} -n{size} -q{num_groups} -p{",".join([str(x) for x in na])} -c{",".join([str(x) for x in cab])} -M {target_path} -v-1'
        elif infermode == 1:
            command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -M {target_path}'
        elif infermode == 2:
            learn_conv_crit = 1e-6 if learn_conv_crit is None else learn_conv_crit
            learn_max_time = 1000 if learn_max_time is None else learn_max_time
            init_epsilon = 0.1 if init_epsilon is None else init_epsilon
            command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -P{init_epsilon},{np.around(d, 2)} -M {target_path} -v-1 -E{learn_conv_crit} -T{learn_max_time}'
        elif infermode == 3:
            command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -p{",".join([str(x) for x in na])} -c{",".join([str(x) for x in cab])} -M {target_path}'
        elif infermode == 4 or infermode == 5 or infermode == 6:
            learnq_path = f'./other/mode_net/data/learnq{str(processId) if processId is not None else ""}.txt'
            learnq_command = f'./other/mode_net/sbm learnq -l {net_gml_path} -n{size} -P0,{d} -M {learnq_path} -v-1'
            print(learnq_command)
            os.system(learnq_command)
            with open(learnq_path, 'rb') as f:
                offset = -50
                while True:
                    f.seek(offset, 2)
                    lines = f.readlines()
                    if len(lines) >= 2:
                        last_line = lines[-1]
                        num_groups = int(last_line.strip())
                        break
                    offset *= 2
            if infermode == 4:
                command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -M {target_path} -v-1'
            elif infermode == 5:
                command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -P0.1,{np.around(d, 2)} -M {target_path} -v-1'
            elif infermode == 6:
                command = f'./other/mode_net/sbm learn -l {net_gml_path} -n{size} -q{num_groups} -p{",".join([str(x) for x in na])} -c{",".join([str(x) for x in cab])} -M {target_path} -v-1'
            else:
                command = f''
        else:
            command = f''
        print(command)
        os.system(command)
        with open(target_path, 'r') as f:
            line = f.readline()
            if len(line) != 0:
                free_energy = float(line.split('\t')[0].split('=')[1])
                f.readline()
                line = f.readline()
                partition = np.array([int(i) for i in line.strip().split(' ')])
            else:
                print("Some error happend with BP code(Maybe Segmentation fault)")
                free_energy = np.nan
                partition = np.array([])
        return partition, free_energy

    def BP_MDL_learnq(self, groupId, max_q=6, processId=None, rhodelta="", init_epsilon=None, learn_conv_crit=None, learn_max_time=None):
        g = nx.from_scipy_sparse_array(self.A)
        N = g.number_of_nodes()
        E = g.number_of_edges()
        learnq_path = f'./other/mode_net/data/BPlearnq_MDL_FreeEnergy_{str(processId) if processId is not None else ""}.txt'
        with open(learnq_path, 'a') as fw:
                fw.write(rhodelta + '\n')
        last_partition = None
        last_q = 0
        last_MDL = None
        for q in range(1, max_q):
            if q > 1:
                partition, _ = self.BP(num_groups=q, na=None, cab=None, groupId=groupId, processId=processId, infermode=2, init_epsilon=init_epsilon, learn_conv_crit=learn_conv_crit, learn_max_time=learn_max_time)
            else:
                partition = np.array([0 for i in range(N)])
                # f = "No calculate"
            if np.size(partition) != 0:
                q_partition = np.size(np.unique(partition))
                Epsilonb = self.desc_length(self.A, q, partition)
                # x = q * (q + 1) / (2*E)
                # hx = ((1+x)*np.log(1+x)-x*np.log(x))
                # Lt = E * hx + N * np.log(q)
                # It = 0
                # unique_partition = np.unique(partition)
                # # St = E
                # for r in unique_partition:
                #     for s in unique_partition:
                #         r_index = np.where(partition==r)[0]
                #         s_index = np.where(partition==s)[0]
                #         n_r = np.size(r_index)
                #         n_s = np.size(s_index)
                #         ers = np.sum(self.A[np.ix_(r_index, s_index)])
                #         # ers = ers if r != s else ers/2
                #         # St -= 1 / 2 * ers * np.log(ers / (n_r * n_s))
                #         mrs = ers / (2*E)
                #         wr = n_r / N
                #         ws = n_s / N
                #         It += mrs * np.log(mrs / (wr * ws)) if mrs != 0 else 0
                # Epsilonb = Lt - E * It
                # Epsilont = Lt + St
                log = f'q={q}, q_partition={q_partition}, MDL={Epsilonb}'
            else:
                log = f'q={q}, Error in BP code (Unknown reason)'
            print(log)
            with open(learnq_path, 'a') as fw:
                fw.write(log + '\n')
            if np.size(partition) != 0:
                if q == 1:
                    last_partition = partition
                    last_q = 1
                    last_MDL = Epsilonb
                elif Epsilonb >= last_MDL:
                    return last_partition, last_q
                else:
                    last_partition = partition
                    last_q = q
                    last_MDL = Epsilonb
        num_group = np.size(np.unique(last_partition))
        return last_partition, num_group
    
    def BP_FE_learnq(self, groupId, processId=None, strId="", init_epsilon=None, learn_conv_crit=None, learn_max_time=None, 
                        learn_time_forq=20, max_learn_q=5, stop_when_increasing_f=True):
        # Initial g and d
        g = nx.from_scipy_sparse_array(self.A)
        size = g.number_of_nodes()
        '''Drop the weights from a networkx weighted graph.'''
        for node, edges in nx.to_dict_of_dicts(g).items():
            for edge, attrs in edges.items():
                attrs.pop('weight', None)
            g.nodes[node].pop('label', None)
            g.nodes[node]['value'] = str(groupId[node])
        d = self.A.sum() / self.A.shape[0]
        self.g = g
        self.d = d
        self.n = size
        learnq_path = f'./other/mode_net/record_f/{strId}_BPlearnq_FreeEnergy_{str(processId) if processId is not None else ""}_inite{init_epsilon}.txt'
        # with open(learnq_path, 'a') as fw:
        #         fw.write(rhodelta + '\n')
        last_partition = None
        last_q = 0
        last_FE = None
        result_partition = None
        result_num_group = 0
        for q in range(1, max_learn_q+1):
            if q > 1:
                minf_partition = None
                min_f = None
                for t in range(learn_time_forq):
                    print(f'Time {t} Initialization for BP learnq by FE!')
                    partition, f = self.BP(num_groups=q, na=None, cab=None, groupId=groupId, processId=processId, infermode=2, init_epsilon=init_epsilon, learn_conv_crit=learn_conv_crit, learn_max_time=learn_max_time)
                    if min_f is None:
                        minf_partition = partition
                        min_f = f
                    elif f < min_f:
                        minf_partition = partition
                        min_f = f
                    else:
                        pass
                    log = f'q={q} t={t} f={f} min_f={min_f} detect_q={np.size(np.unique(partition))}\n'
                    with open(learnq_path, 'a') as fw:
                        fw.write(log)
                partition = minf_partition
                f = min_f
            else:
                # q = 1 only 1 time is enough
                partition, f = self.BP(num_groups=q, na=None, cab=None, groupId=groupId, processId=processId, infermode=2, init_epsilon=init_epsilon, learn_conv_crit=learn_conv_crit, learn_max_time=learn_max_time)
                log = f'q={q} t={1} f={f} min_f={f} detect_q={np.size(np.unique(partition))}\n'
                with open(learnq_path, 'a') as fw:
                    fw.write(log)
            # if np.size(partition) != 0:
            #     q_partition = np.size(np.unique(partition))
            #     log = f'Final select q={q}, q_partition={q_partition}, FreeEnergy={f}'
            # else:
            #     log = f'Final select q={q}, Error in BP code (Unknown reason)'
            if np.size(partition) != 0:
                if q == 1:
                    last_partition = partition
                    last_q = 1
                    last_FE = f
                elif f >= last_FE and result_partition is None and result_num_group == 0:
                    # Find first nondecreasing q and record the result
                    result_partition = last_partition
                    num_group = np.size(np.unique(last_partition))
                    result_num_group = num_group
                    if stop_when_increasing_f:
                        break
                    else:
                        last_partition = partition
                        last_q = q
                        last_FE = f
                        continue
                else:
                    last_partition = partition
                    last_q = q
                    last_FE = f
        if result_partition is None and result_num_group == 0:
            result_partition = last_partition
            num_group = np.size(np.unique(last_partition))
            result_num_group = num_group
        log = f'Finial result number of group {result_num_group}\n'
        with open(learnq_path, 'a') as fw:
            fw.write(log)
        return result_partition, result_num_group
    
    def BH_MDL_learnq(self, max_q=6, processId=None, rhodelta=""):
        g = nx.from_scipy_sparse_array(self.A)
        N = g.number_of_nodes()
        last_partition = None
        last_q = 0
        last_MDL = None
        for q in range(1, max_q):
            if q > 1:
                partition, _ = self.BetheHessian(num_groups=q)
            else:
                partition = np.array([0 for i in range(N)])
                # f = "No calculate"
            if np.size(partition) != 0:
                q_partition = np.size(np.unique(partition))
                Epsilonb = self.desc_length(self.A, q, partition)
                # x = q * (q + 1) / (2*E)
                # hx = ((1+x)*np.log(1+x)-x*np.log(x))
                # Lt = E * hx + N * np.log(q)
                # It = 0
                # unique_partition = np.unique(partition)
                # # St = E
                # for r in unique_partition:
                #     for s in unique_partition:
                #         r_index = np.where(partition==r)[0]
                #         s_index = np.where(partition==s)[0]
                #         n_r = np.size(r_index)
                #         n_s = np.size(s_index)
                #         ers = np.sum(self.A[np.ix_(r_index, s_index)])
                #         # ers = ers if r != s else ers/2
                #         # St -= 1 / 2 * ers * np.log(ers / (n_r * n_s))
                #         mrs = ers / (2*E)
                #         wr = n_r / N
                #         ws = n_s / N
                #         It += mrs * np.log(mrs / (wr * ws)) if mrs != 0 else 0
                # Epsilonb = Lt - E * It
                # Epsilont = Lt + St
                log = f'q={q}, q_partition={q_partition}, MDL={Epsilonb}'
            else:
                log = f'q={q}, Error in BH code (Unknown reason)'
            print(log)
            if np.size(partition) != 0:
                if q == 1:
                    last_partition = partition
                    last_q = 1
                    last_MDL = Epsilonb
                elif Epsilonb >= last_MDL:
                    return last_partition, last_q
                else:
                    last_partition = partition
                    last_q = q
                    last_MDL = Epsilonb
        num_group = np.size(np.unique(last_partition))
        return last_partition, num_group

    @staticmethod
    def desc_length(A, q, partition):
        N = np.shape(A)[0]
        E = np.sum(A) / 2
        x = q * (q + 1) / (2 * E)
        hx = ((1 + x) * np.log(1 + x) - x * np.log(x))
        Lt = E * hx + N * np.log(q)
        It = 0
        unique_partition = np.unique(partition)
        # St = E
        for r in unique_partition:
            for s in unique_partition:
                r_index = np.where(partition == r)[0]
                s_index = np.where(partition == s)[0]
                n_r = np.size(r_index)
                n_s = np.size(s_index)
                ers = np.sum(A[np.ix_(r_index, s_index)])
                # St -= 1 / 2 * ers * np.log(ers / (n_r * n_s))
                mrs = ers / (2 * E)
                wr = n_r / N
                ws = n_s / N
                It += mrs * np.log(mrs / (wr * ws)) if mrs != 0 else 0
        Epsilonb = Lt - E * It  # equation (6) in "Parsimonious module inference in large networks"
        return Epsilonb


    def MDL(self, processId=None):
        # construct g
        g = gt.Graph(directed=False)
        x, y, _ = sparse.find(self.A == 1)
        edges = [(x[i], y[i]) for i in range(len(x))]
        g.add_edge_list(edges)
        n = g.num_vertices()
        state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False), multilevel_mcmc_args=dict(B_max=10))
        b = state.get_blocks()
        partition = np.array([b[i] for i in range(n)])
        q = np.size(np.unique(partition))
        return partition, q


def test_main():
    # TEST 1
    # fileID = 'amiExp4.20'
    # load_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    # plot_rhos, plot_zs, full_ami, sub_ami = CommunityDetect.read_exp(load_path=load_path)
    # import _FigureJiazeHelper
    # _FigureJiazeHelper.color_scatter_2d(plot_rhos, plot_zs, full_ami, title="AMI for full graph", xlabel=r'$\rho$',
    #                                     ylabel=r'$z$', save_path=None)
    # TEST 0
    rho = 0.1
    X = 2  # Number of Meta
    Z = 3  # Number of Group in each Meta
    n = X * Z * 2000
    d = 50
    delta = 0.01
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout

    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    A = msbm.sample()
    cd = CommunityDetect(A)
    print(np.unique(msbm.groupId, return_counts=True))
    start = time.time()
    DCBHpartition, numgroups, zetas = cd.DCBetheHessian()
    # print("FULL, real labels:", np.unique(msbm.groupId))
    # print("FULL, detect labels:", np.unique(DCBHpartition))
    print("FULL, DCBH result:", adjusted_mutual_info_score(msbm.groupId, DCBHpartition), f"time={time.time()-start}")
    start = time.time()
    BHpartition, numgroups = cd.BetheHessian()
    print("FULL, BH result:", adjusted_mutual_info_score(msbm.groupId, BHpartition), f"time={time.time()-start}")


    filterA, filterGroupId = msbm.filter(A, metaId=0)
    print(np.unique(filterGroupId, return_counts=True))
    # # print("SUB graph size:", np.size(filterGroupId))
    # # print("SUB, real labels:", np.unique(filterGroupId0))
    cd = CommunityDetect(filterA)
    start = time.time()
    DCBHpartition, numgroups, zetas = cd.DCBetheHessian()
    # print("SUB, real labels:", np.unique(filterGroupId))
    # print("SUB, detect labels:", np.unique(DCBHpartition))
    print("SUB, DCBH result:", adjusted_mutual_info_score(filterGroupId, DCBHpartition), f"time={time.time()-start}")
    # # print("SUB graph size:", np.size(filterGroupId))
    start = time.time()
    BHpartition, numgroups = cd.BetheHessian()
    print("SUB, BH result:", adjusted_mutual_info_score(filterGroupId, BHpartition), f"time={time.time()-start}")

def test_main_BP():
    pid = os.getpid()
    rho = 0.24
    X = 2  # Number of Meta
    Z = 3  # Number of Group in each Meta
    n = X * Z * 2000
    d = 50
    delta = 0.01
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 1e-5 if pin < 1e-5 else pin
    pout = 1e-5 if pout < 1e-5 else pout
    start = time.time()
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    A = msbm.sample()
    cd = CommunityDetect(A)
    na = [1/(X*Z)] * (X*Z)
    cab = []
    for x in np.nditer(msbm.ps):
        cab.append(np.around(x*n, 2))
    partition = cd.BP(X*Z, na, cab, msbm.groupId, processId=str(pid)+"FULL")
    print("FULL, BP result:", adjusted_mutual_info_score(msbm.groupId, partition), f"time={time.time()-start}")
    partition, _ = cd.BetheHessian()
    print("FULL, BH result:", adjusted_mutual_info_score(msbm.groupId, partition), f"time={time.time()-start}")
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    sub_num_groups_given = np.size(np.unique(filterGroupId))
    subsize = np.size(filterGroupId)
    v, counts = np.unique(filterGroupId, return_counts=True)
    na = counts / np.size(filterGroupId)
    cab = []
    for x in np.nditer(msbm.ps[v, :][:, v]):
        cab.append(np.around(x*subsize, 2))
    subpartition = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId, processId=str(pid)+"SUB")
    print("SUB, BP result:", adjusted_mutual_info_score(filterGroupId, subpartition), f"time={time.time()-start}")
    subpartition, _ = CommunityDetect(filterA).BetheHessian()
    print("SUB, BH result:", adjusted_mutual_info_score(filterGroupId, subpartition), f"time={time.time()-start}")

def test_main_MDLlearnq():
    pid = os.getpid()
    rho = 0.1
    X = 2  # Number of Meta
    Z = 3  # Number of Group in each Meta
    n = X * Z * 2000
    d = 50
    delta = 0.01
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 1e-5 if pin < 1e-5 else pin
    pout = 1e-5 if pout < 1e-5 else pout
    start = time.time()
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    A = msbm.sample()
    cd = CommunityDetect(A)
    fpartition, fq = cd.BP_MDL_learnq(groupId=msbm.groupId, processId=str(pid)+"FULL")
    half_time = time.time()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    cd = CommunityDetect(filterA)
    spartition, sq = cd.BP_MDL_learnq(groupId=filterGroupId, processId=str(pid)+"SUB")
    print(f"Sub Time: {time.time() - half_time}, Full Time: {half_time-start}")
    print("FULL, BP result:", adjusted_mutual_info_score(msbm.groupId, fpartition), f"# of groups:{fq}")
    print("SUB, BP result:", adjusted_mutual_info_score(filterGroupId, spartition), f"# of groups:{sq}")

def test_main_MDL():
    pid = os.getpid()
    rho = 0.1
    X = 2  # Number of Meta
    Z = 3  # Number of Group in each Meta
    n = X * Z * 2000
    d = 50
    delta = 0.01
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 1e-5 if pin < 1e-5 else pin
    pout = 1e-5 if pout < 1e-5 else pout
    start = time.time()
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    A = msbm.sample()
    cd = CommunityDetect(A)
    fpartition, fq = cd.MDL(processId=str(pid)+"FULL")
    half_time = time.time()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    cd = CommunityDetect(filterA)
    spartition, sq = cd.MDL(processId=str(pid)+"SUB")
    print(f"Sub Time: {time.time() - half_time}, Full Time: {half_time-start}")
    print("FULL, BP result:", adjusted_mutual_info_score(msbm.groupId, fpartition), f"# of groups:{fq}")
    print("SUB, BP result:", adjusted_mutual_info_score(filterGroupId, spartition), f"# of groups:{sq}")


if __name__ == '__main__':
    # test_main()
    # test_main_BP()
    # test_main_MDLlearnq()
    test_main_MDL()
