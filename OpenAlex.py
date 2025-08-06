import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import csr_array
import pickle
from itertools import chain
import pyalex
from pyalex.api import OpenAlexResponseList
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders



class Alex:
    def __init__(self, works=None, load_path=None):
        self.author_n = 0
        self.paper_n = 0
        self.incidence_H = None  # authors * papers
        self.citation_A = None  # papers * papers
        self.works_idmap = dict()  # id -> openalexid. for following meta checking
        self.authors_idmap = dict()  # id -> openalexid. for following meta checking
        self.works_info = dict()  # id -> info
        self.authors_info = dict()  # id -> info
        if load_path is None:
            self.construct(works)
        else:
            self.load(load_path)
        pass

    @staticmethod
    def works_from_seed(seed_works=["W3086152724"], n_step=1):
        """
        Get works from seed works, n_step times.
        :param seed_works: a list of OpenAlex works ID, should be your interest seed works
        :param n_step: number of steps to expand in collaboration network
        :return: a list of OpenAlex IDs, and a list of pyalex Works objects
        """
        alex_works = [Works()[w] for w in seed_works]  # Get works from OpenAlex IDs
        alex_works_id = set(seed_works)
        authors = set()
        # authors = set(["A" + author['author']['id'].split("/A")[1] for author in seed_works['authorships']])
        traverse_works = alex_works.copy()  # Start with the seed works
        for i_step in range(n_step):
            print(f"Step {i_step}, number_works: {len(traverse_works)}, ...", end='\n')
            new_alex_works = []
            new_alex_works_id = set()
            for work in tqdm(traverse_works, desc=f'Traversing works in {i_step} step...'):
                authors_alexid = ["A" + author['author']['id'].split("/A")[1] for author in work['authorships']]
                for author_alexid in authors_alexid:
                    if author_alexid not in authors:
                        authors.add(author_alexid)
                        # get works for this author
                        # filter_existed_works = "+".join(["!"+w for w in alex_works_id])
                        # print(filter_existed_works)
                        works_for_author = Works().filter(authorships={"author": {"id": author_alexid}})  # , openalex=filter_existed_works)
                        for w in chain(*works_for_author.paginate(per_page=200, n_max=None)):
                            w_openalex_id = "W" + w['id'].split('/W')[1]
                            if w_openalex_id not in alex_works_id:
                                new_alex_works_id.add(w_openalex_id)
                                new_alex_works.append(w)
                                alex_works.append(w)
                                alex_works_id.add(w_openalex_id)
                    # print(f"Finished author {author_alexid}, works found: {len(new_alex_works)}")
            traverse_works = new_alex_works.copy()  # Update the works to traverse in the next step
        return list(alex_works_id), alex_works

    def load_all_works(self, works):
        if works is None:
            works = Works().filter(publication_year=2000,
                                   language='en',
                                   is_oa=True,
                                   authors_count='>1',
                                   concepts_count='>0',
                                   has_references=True,
                                   primary_topic={"domain":{'id': '2'}})  # 1 life, 2 social, 3 physical, 4 health
        if isinstance(works, pyalex.api.Works):
            print(works.get().meta)
            all_works = chain(*works.paginate(per_page=200, n_max=None))
        elif isinstance(works, list) and all(isinstance(w, str) for w in works):
            # If works is a list of OpenAlex IDs
            print(f"Loading {len(works)} works from OpenAlex IDs...")
            all_works = [Works()[alexid] for alexid in tqdm(works)]
        elif isinstance(works, list) and all(isinstance(w, pyalex.api.Work) for w in works):
            print(f"Loading {len(works)} works from OpenAlex objects...")
            all_works = works
        return all_works
    
    def construct(self, works=None):
        data = []
        row_ind = []
        col_ind = []
        works_idmap = dict()  # openalexid -> id
        authors_idmap = dict()
        # Iterate all works
        all_works = self.load_all_works(works)
        # print(len(all_works))
        # total_len = sum([1 for w in all_works])
        for work in tqdm(all_works, desc='Loading works...'):
            # Test for 1-time get
            # works = works.get()
            # for work in works:
            # find work and authors id, add to map dict, add authorship in indicence matrix
            work_openalex_id = work['id'].split('/W')[1]
            authors_openalex_id = [author['author']['id'].split('/A')[1] for author in work['authorships']]
            if work_openalex_id not in works_idmap.keys():
                works_idmap[work_openalex_id] = self.paper_n
                self.paper_n += 1
            for author_openalex_id in authors_openalex_id:
                if author_openalex_id not in authors_idmap.keys():
                    authors_idmap[author_openalex_id] = self.author_n
                    self.author_n += 1
                row_ind.append(authors_idmap[author_openalex_id])
                col_ind.append(works_idmap[work_openalex_id])
                data.append(1)
            for author in work['authorships']:
                author_openalex_id = author['author']['id'].split('/A')[1]
                author_id = authors_idmap[author_openalex_id]
                # add author other info: name
                if author_id not in self.authors_info.keys():
                    self.authors_info[author_id] = dict()
                    self.authors_info[author_id]['name'] = author['author']['display_name']
            # add work other info: title, citations, topics, source(journal), etc.
            if works_idmap[work_openalex_id] not in self.works_info.keys():
                self.works_info[works_idmap[work_openalex_id]] = dict()
                self.works_info[works_idmap[work_openalex_id]]['title'] = work['title']
                self.works_info[works_idmap[work_openalex_id]]['citations'] = []
                for refer_work in work['referenced_works']:
                    refer_work_id = refer_work.split('/W')[1]
                    self.works_info[works_idmap[work_openalex_id]]['citations'].append(refer_work_id)
                if 'primary_topic' in work.keys() and work['primary_topic'] is not None:
                    self.works_info[works_idmap[work_openalex_id]]['topic'] = \
                        (work['primary_topic']['id'].split('/T')[1], \
                         work['primary_topic']['display_name'])
                    self.works_info[works_idmap[work_openalex_id]]['subfield'] = \
                        (work['primary_topic']['subfield']['id'].split('subfields/')[1], \
                         work['primary_topic']['subfield']['display_name'])
                    self.works_info[works_idmap[work_openalex_id]]['field'] = \
                        (work['primary_topic']['field']['id'].split('fields/')[1], \
                         work['primary_topic']['field']['display_name'])
                    self.works_info[works_idmap[work_openalex_id]]['domain'] = \
                        (work['primary_topic']['domain']['id'].split('domains/')[1], \
                         work['primary_topic']['domain']['display_name'])
                else:
                    self.works_info[works_idmap[work_openalex_id]]['topic'] = (None, None)
                    self.works_info[works_idmap[work_openalex_id]]['subfield'] = (None, None)
                    self.works_info[works_idmap[work_openalex_id]]['field'] = (None, None)
                    self.works_info[works_idmap[work_openalex_id]]['domain'] = (None, None)
                if "primary_location" in work.keys() and work['primary_location'] is not None and work['primary_location']['source'] is not None:
                    self.works_info[works_idmap[work_openalex_id]]['source'] = \
                        (work['primary_location']['source']['id'].split('/S')[1], work['primary_location']['source']['display_name'])
                else:
                    self.works_info[works_idmap[work_openalex_id]]['source'] = (None, None)
        # print(len(data))
        self.incidence_H = csr_array((data, (row_ind, col_ind)))
        self.works_idmap = dict({works_idmap[k]:k for k in works_idmap.keys()})  # reverse map
        self.authors_idmap = dict({authors_idmap[k]:k for k in authors_idmap.keys()})  # reverse map
        self.summary()
        pass
    
    def summary(self):
        print(f"Number of Author {self.author_n}, Number of Paper {self.paper_n} \n", 
                  f"Average #_papers per author {self.incidence_H.sum() / self.author_n} \n", 
                  f"Average #_coauthors per paper {self.incidence_H.sum() / self.paper_n}")
        topics = set()
        for i in self.works_info:
            if self.works_info[i]['topic'][0] is not None:
                topics.add(self.works_info[i]['topic'][0])
        print(f"Number of topics {len(topics)}.")
        num_citation = []
        for i in self.works_info:
            num_citation.append(len(self.works_info[i]['citations']))
        print(f'Most cite {max(num_citation)} works, least cite {min(num_citation)} works.')
    
    def construct_citation_matrix(self):
        data = []
        row_ind = []
        col_ind = []
        works_alexid2id_map = dict({self.works_idmap[k]:k for k in self.works_idmap.keys()})
        # Iterate all works
        for i in tqdm(range(self.paper_n), desc='Constructing citation matrix...'):
            work_id = i
            if work_id in self.works_info.keys():
                citations_alexid = self.works_info[work_id]['citations']
                for citation_alexid in citations_alexid:
                    if citation_alexid in works_alexid2id_map.keys():
                        j = works_alexid2id_map[citation_alexid]
                        row_ind.append(i)
                        col_ind.append(j)
                        data.append(1)
        self.citation_A = csr_array((data, (row_ind, col_ind)), shape=(self.paper_n, self.paper_n))
    
    def construct_JJcitation_matrix(self):
        """ 
        Construct journal-journal citation matrix.
        """
        journals_alexid2id_map = dict()
        works_alexid2id_map = dict({self.works_idmap[k]:k for k in self.works_idmap.keys()})
        data = []
        row_ind = []
        col_ind = []
        for i in tqdm(range(self.paper_n), desc='Constructing journal-journal citation matrix...'):
            work_id = i
            if work_id in self.works_info.keys():
                source_alexid = self.works_info[work_id]['source'][0]
                if source_alexid is not None:
                    if source_alexid not in journals_alexid2id_map.keys():
                        journals_alexid2id_map[source_alexid] = len(journals_alexid2id_map)
                    j = journals_alexid2id_map[source_alexid]
                    citations_alexid = self.works_info[work_id]['citations']
                    for citation_alexid in citations_alexid:
                        if citation_alexid in works_alexid2id_map.keys():
                            citation_work_id = works_alexid2id_map[citation_alexid]
                            if citation_work_id in self.works_info.keys():
                                citation_source_alexid = self.works_info[citation_work_id]['source'][0]
                                if citation_source_alexid is not None:
                                    if citation_source_alexid not in journals_alexid2id_map.keys():
                                        journals_alexid2id_map[citation_source_alexid] = len(journals_alexid2id_map)
                                    k = journals_alexid2id_map[citation_source_alexid]
                                    row_ind.append(j)
                                    col_ind.append(k)
                                    data.append(1)
        JJN_A = csr_array((data, (row_ind, col_ind)), shape=(len(journals_alexid2id_map), len(journals_alexid2id_map)))
        journals_id2alexid_map = dict({v: k for k, v in journals_alexid2id_map.items()})  # reverse map
        return JJN_A, journals_id2alexid_map

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.author_n, self.paper_n, self.incidence_H, 
                         self.citation_A, self.works_idmap, self.authors_idmap, self.works_info, self.authors_info], f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            attributes = pickle.load(f)
            self.author_n = attributes[0]
            self.paper_n = attributes[1]
            self.incidence_H = attributes[2]
            self.citation_A = attributes[3]
            self.works_idmap = attributes[4]
            self.authors_idmap = attributes[5]
            self.works_info = attributes[6]
            self.authors_info = attributes[7]


if __name__ == "__main__":
    alex_works_id, alex_works = Alex.works_from_seed(n_step=2)
