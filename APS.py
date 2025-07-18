import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_array
import matplotlib.pyplot as plt
from _FigureJiazeHelper import *
from _HyperSBM import *
from _HyperCommunityDetection import *
import pickle
import warnings
import pandas
import time
import hypernetx as hnx

def save_idmap(path, d):
    with open(path, 'w') as f:
        for k in d.keys():
            f.write(f"{k} {d[k]}\n")

            
def load_idmap(path):
    d = dict()
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            d[int(line[0])] = int(line[1])
    return d

class APS:
    def __init__(self):
        self.author_n = 0
        self.paper_n = 0
        self.incidence_H = None  # authors * papers
        self.citation_A = None  # papers * papers
        self.construct()
    
    def construct(self):
        authorship_file = "./aps/data/authorships_afterdisambiguation.csv"
        # Construct authorship incidence matrix (Bipartite or Hyper)
        with open(authorship_file, 'r') as fr:
            fr.readline()
            data = []
            row_ind = []
            col_ind = []
            author_ids = []
            paper_ids = []
            author_papers = set()
            for line in tqdm(fr.readlines(), desc='Load Authorship'):
                line = line.strip().split(',')
                paper_id = int(line[0])
                author_id = int(line[1])
                if (author_id, paper_id) not in author_papers:
                    author_papers.add((author_id, paper_id))
                else:
                    continue
                    # print(f"{(author_id, paper_id)} repeats") # There are some author paper repeat in file
                # if paper_id > 0 and paper_id - col_ind[-1] >= 2:
                #     print(paper_id)
                author_ids.append(author_id)
                paper_ids.append(paper_id)
                data.append(1)
            # check the authoid and paperid in file is not continues
            # print(f"{np.size(np.unique(author_ids))}, {max(author_ids)}, {np.size(np.unique(paper_ids))}, {max(paper_ids)}")
            unique_author_ids = np.unique(author_ids)
            self.author_n = np.size(np.unique(author_ids))
            authorId_map_ind = dict({unique_author_ids[i]:i for i in range(self.author_n)})
            
            # For paper we don't unique that considering the citation part
            # unique_paper_ids = np.unique(paper_ids)
            # self.paper_n = np.size(np.unique(unique_paper_ids))
            # paperId_map_ind = dict({unique_paper_ids[i]:i for i in range(self.paper_n)})
            self.paper_n = np.max(paper_ids)+1
            # TODO save these map for check
            save_authorId_map = "./aps/authorId_map_ind.txt"
            save_idmap(save_authorId_map, authorId_map_ind)
            
            row_ind = [authorId_map_ind[aid] for aid in author_ids]
            col_ind = paper_ids
            self.incidence_H = csr_array((data, (row_ind, col_ind)))
            print(f"Number of Author {self.author_n}, Number of Paper {self.paper_n} \n", 
                  f"Average #_papers per author {self.incidence_H.sum() / self.author_n} \n", 
                  f"Average #_coauthors per paper {self.incidence_H.sum() / self.paper_n} \n")
        
        citation_file = "./aps/data/citations_withID.csv"
        # Construct citation adjacent matrix (Directed network)
        with open(citation_file, 'r') as fr:
            fr.readline()
            data = []
            citing_ids = []
            cited_ids = []
            for line in tqdm(fr.readlines(), desc='Load Citation'):
                line = line.strip().split(',')
                citing_id = int(float(line[2]))
                cited_id = int(float(line[3]))
                citing_ids.append(citing_id)
                cited_ids.append(cited_id)
                data.append(1)
                # Check there are some paper cited but no author
                # if citing_id not in paperId_map_ind.keys() or cited_id not in paperId_map_ind.keys():
                #     print(f"{citing_id}, {cited_id} is not in authorship file")
            # print(f"{np.size(np.unique(citing_ids))}, {max(citing_ids)}, {np.size(np.unique(cited_ids))}, {max(cited_ids)}")
            self.citation_A = csr_array((data, (citing_ids, cited_ids)), shape=(self.paper_n, self.paper_n))
            print(f"Average #_cited_papers per citing_paper {self.citation_A.sum() / self.paper_n} \n")
            
    def papers_info(self, paper_ids, paper_partition, save_path):
        partition_dict = dict(zip(paper_ids, paper_partition))
        
        paper_path = "./aps/meta/publications.csv"
        if save_path is not None:
            paper_info_path = save_path
        else:
            paper_info_path = "./aps/result/paper_partition_info.csv"  # to save
        # Load topics
        paper_topic_path = "./aps/meta/publication_topics.csv"
        paper_topics = dict({i:{0:[], 1:[], 2:[], 'primary_topic(concept)':-1} for i in range(self.paper_n)})
        topic_type = dict({0:"area", 1:"discipline", 2:"concept"})
        with open(paper_topic_path, 'r') as fr:
            fr.readline()
            for line in tqdm(fr.readlines(), desc='Load paper_topics'):
                line = line.strip().split(',')
                paper_topics[int(line[0])][int(line[2])].append(int(line[1]))
                if line[-1] == "True":
                    paper_topics[int(line[0])]['primary_topic(concept)'] = int(line[1])
        # Load topic names
        areas = dict()
        with open("./aps/meta/areas.csv", 'r') as fr:
            fr.readline()
            for line in tqdm(fr.readlines(), desc='Load areas'):
                line = line.strip().split(',')
                areas[int(line[0])] = line[2]
        disciplines = dict()
        with open("./aps/meta/disciplines.csv", 'r') as fr:
            fr.readline()
            for line in tqdm(fr.readlines(), desc='Load disciplines'):
                line = line.strip().split(',')
                disciplines[int(line[0])] = line[2]
        concepts = dict()
        with open("./aps/meta/concepts.csv", 'r') as fr:
            fr.readline()
            for line in tqdm(fr.readlines(), desc='Load concepts'):
                line = line.strip().split(',')
                concepts[int(line[0])] = line[3]
        # Load paper and write
        with open(paper_path, 'r') as fr:
            with open(paper_info_path, 'w') as fw:
                fw.write("id_publication,id_journal,timestamp,doi,areas,disciplines,concepts,primary_concept,partitions\n")
                fr.readline()
                for rline in tqdm(fr.readlines(), desc='Load papers'):
                    wline = rline.strip().split(',')
                    pid = int(wline[0])
                    wline.append(" ".join([str(i) for i in sorted(paper_topics[pid][0])]))
                    wline.append(" ".join([str(i) for i in sorted(paper_topics[pid][1])]))
                    wline.append(" ".join([str(i) for i in sorted(paper_topics[pid][2])]))
                    wline.append(str(paper_topics[pid]['primary_topic(concept)']))
                    if pid in partition_dict.keys():
                        wline.append(str(partition_dict[pid]))
                    else:
                        wline.append("")
                    wline = ",".join(wline)
                    fw.write(wline+"\n")
                    

class HyperPaper(APS):
    def __init__(self):
        super().__init__()
        self.n = self.paper_n
        self.e = self.author_n
        self.H = self.incidence_H.transpose()
        
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        print(f"There are {order_count[0]} authors with 1 paper: number of order_1 hyperedge. To be removed!")
        self.e = self.e - order_count[0]
        nonorder1_column = (self.H.sum(axis=0).flatten()!=1).nonzero()
        self.H = (self.H.tocsc()[:, nonorder1_column[0]]).tocsr()
        # print(np.unique(self.H.data, return_counts=True))
        
        node_degree, degree_count = np.unique(self.H.sum(axis=1).flatten(),  return_counts=True)
        print(f"There are {degree_count[0]} papers with 0 author: number of degree_0 nodes. To be removed!")
        self.n = self.n - degree_count[0]
        self.nondegree0_row = (self.H.sum(axis=1).flatten()!=0).nonzero()
        self.nodes = self.nondegree0_row[0]
        self.H = self.H[self.nondegree0_row[0], :]
        
        # self.max_component()
        
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        self.Ks = edge_order
        self.summary()
    
    def summary(self):
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        node_degree, degree_count = np.unique(self.H.sum(axis=1).flatten(),  return_counts=True)
        print(f"Hypergraph on papers, each author is a hyperedge. n={self.n}, e={self.e}\n Ks & #hyperedge per order={np.array([edge_order, order_count])}")
        print(f"Ds & #nodes per degree={np.array([node_degree, degree_count])}")
    
    
    def max_component(self, connected_components=None):
        if connected_components is None:
            hyper_g = hnx.Hypergraph.from_incidence_matrix(self.H)
            connected_components = list(hyper_g.s_connected_components(edges=False))
        print(f"There are {len(connected_components)} connected components now, we select the maximum one")
        max_component = np.array(list(connected_components[0]))
        self.H = self.H[max_component, :]  # Directly cut the nodes in max component
        self.nodes = self.nondegree0_row[0][max_component]
        self.n = np.size(max_component)
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        print(f"There are {order_count[0]} authors with {edge_order[0]} paper after select max component: To be removed!")
        self.e = self.e - order_count[0]
        nonorder0_column = (self.H.sum(axis=0).flatten()!=0).nonzero()
        self.H = (self.H.tocsc()[:, nonorder0_column[0]]).tocsr()
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        node_degree, degree_count = np.unique(self.H.sum(axis=1).flatten(),  return_counts=True)
        self.Ks = edge_order
        self.summary()
        
    def get_operator(self, operator='BH', r=0, consider_ks=None):
        if operator == "BH":
            edge_order = self.H.sum(axis=0).flatten()
            D = None
            A = None
            self.H = self.H.tocsc()
            if consider_ks is None:
                Ks = self.Ks
            else:
                Ks = consider_ks
            for k in tqdm(Ks, desc='Constructing HyperBH...'):
                edge_index = np.where(edge_order == k)[0]
                Hk = self.H[:, edge_index]
                Dk = diags(Hk.sum(axis=1).flatten().astype(float))
                Ak = Hk.dot(Hk.T) - diags(Hk.dot(Hk.T).diagonal())
                if D is None:
                    D = (k-1)/((1-r)*(r+k-1))*Dk
                else:
                    D += (k-1)/((1-r)*(r+k-1))*Dk
                if A is None:
                    A = r/((1-r)*(r+k-1))*Ak
                else:
                    A += r/((1-r)*(r+k-1))*Ak
            B = eye(D.shape[0]) - D + A
            return B
        elif operator == "binary_proj_L":
            projA = self.H.dot(self.H.T)
            projA = projA - diags(projA.diagonal())
            projA[projA > 0] = 1
            D = diags(projA.sum(axis=0))
            L = D - projA
            print(L[:10, :10])
            return L

def cd_apshyperpaper(aps_hyper_paper, given_num_groups=3, consider_ks=None, save_path=None, BHposOperator=None):
    partition_vec, num_groups = HyperCommunityDetect().BetheHessian(aps_hyper_paper, 
                                                                    num_groups=given_num_groups,
                                                                    consider_ks=consider_ks,
                                                                    only_assortative=True, 
                                                                    BHposOperator=BHposOperator)
    if save_path is None:
        save_path = f"./result/partition_paper_hyperBH_given{given_num_groups}Groups_consider{consider_ks}Orders.pkl"
    with open(save_path, "wb") as fw:
            pickle.dump(partition_vec, fw)
    return partition_vec


def main():
    # Construct hypergraph
    aps_hyper_paper = HyperPaper()
    start = time.time()
    hyper_g = hnx.Hypergraph.from_incidence_matrix(aps_hyper_paper.H)
    connected_components = list(hyper_g.s_connected_components(edges=False))
    print(f"Find connected components take {time.time()-start}")
    aps_hyper_paper.max_component(connected_components=connected_components)
    
    # Run CD
    start = time.time()
    given_num_groups = 168 * 2 # 336
    consider_ks = None
    print(f"Consider Orders: {consider_ks}")
    save_path = f"./result/partition_paper_hyperBH_maxcomponent_given{given_num_groups}Groups_considerOrders{consider_ks}.pkl"
    partition_vec = cd_apshyperpaper(aps_hyper_paper, given_num_groups=given_num_groups, consider_ks=consider_ks, 
                                     save_path=save_path)
    print(f"Time cost {time.time()-start}")
    print(np.unique(partition_vec, return_counts=True))
    
    # Save all
    save_path = f"./result/partition_paper_hyperBH_maxcomponent_given{given_num_groups}Groups_considerOrders{consider_ks}.csv"
    aps_hyper_paper.papers_info(aps_hyper_paper.nodes, partition_vec, save_path=save_path)
    
if __name__ == "__main__":
    main()
