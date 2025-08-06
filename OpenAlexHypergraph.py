import hypernetx as hnx
from OpenAlex import *
from _HyperSBM import *
from _HyperCommunityDetection import *

class HyperPaper:
    def __init__(self, alex):
        self.n = alex.paper_n
        self.e = alex.author_n
        self.H = alex.incidence_H.transpose()
        
        edge_order, order_count = np.unique(self.H.sum(axis=0).flatten(), return_counts=True)
        print(f"There are {order_count[0]} authors with 1 paper: number of order_1 hyperedge. To be removed!")
        self.e = self.e - order_count[0]
        self.nonorder1_column = (self.H.sum(axis=0).flatten()!=1).nonzero()
        self.H = (self.H.tocsc()[:, self.nonorder1_column[0]]).tocsr()
        # print(np.unique(self.H.data, return_counts=True))
        
        node_degree, degree_count = np.unique(self.H.sum(axis=1).flatten(),  return_counts=True)
        print(f"There are {degree_count[0] if node_degree[0]==0 else 0} papers with 0 author: number of degree_0 nodes. To be removed!")
        if node_degree[0]==0:
            self.n = self.n - degree_count[0]
        else:
            self.n = self.n - 0
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
            hyper_g = hnx.Hypergraph.from_incidence_matrix(self.H.toarray())
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