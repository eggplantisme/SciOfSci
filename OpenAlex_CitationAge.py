from OpenAlex import *
import os
from multiprocessing import Pool, Manager
import requests

def print_error(value):
    print(value)

def write_results(arg):
    """
    :param arg: citation_ages, _id, save_path, ref_year_map, ref_year_savepath
    :return:
    """
    # save citation ages
    if arg[2] is not None:
        with open(arg[2], 'a') as fw:
            fw.write(f'{arg[1]} {" ".join([str(age) for age in arg[0]])}\n')
    # save ref year 
    if os.path.exists(arg[4]):
        with open(arg[4], 'rb') as fb:
            old_ref_year_map = pickle.load(fb)
            old_ref_year_map.update(arg[3])
            new_ref_year_map = old_ref_year_map
    else:
        new_ref_year_map = arg[3]
    with open(arg[4], 'wb') as fb:
        pickle.dump(arg[3], fb)

def fetch_age(ref_alexid, year, citation_ages, share_ref_year_map, ):
    ref_work = Works()["W"+ ref_alexid]
    ref_year = ref_work["publication_year"]
    share_ref_year_map[ref_alexid] = ref_year

def fetch_citation_age(alex, _id, year, save_path, ref_year_savepath, multiprocess):
    if os.path.exists(ref_year_savepath):
        with open(ref_year_savepath, 'rb') as fb:
            ref_year_map = pickle.load(fb)
            # print(f"Load Ref Paper year map!")
    else:
        ref_year_map = dict()  # {ref_paper_alexid: year}
    
    if multiprocess:
        print(f"Multiprocessing {len(alex.works_info[_id]['citations'])} Citations for {_id} in year {year}!")
        citation_ages = Manager().list()
        share_ref_year_map = Manager().dict(ref_year_map)
        pool = Pool(16)
        for ref_alexid in alex.works_info[_id]['citations']:
            if ref_alexid in ref_year_map.keys():
                ref_year = ref_year_map[ref_alexid]
            else:
                pool.apply_async(fetch_age, args=(ref_alexid, year, citation_ages, share_ref_year_map, ), error_callback=print_error)
        pool.close()
        pool.join()
        ref_year_map = dict(share_ref_year_map)
        citation_ages = list(citation_ages)
        return citation_ages, _id, save_path, ref_year_map, ref_year_savepath
    else:
        citation_ages = []
        for ref_alexid in tqdm(alex.works_info[_id]['citations'], desc=f"Citations({len(alex.works_info[_id]['citations'])}) for {_id}"):
            if ref_alexid in ref_year_map.keys():
                ref_year = ref_year_map[ref_alexid]
            else:
                try:
                    ref_work = Works()["W"+ ref_alexid]
                except requests.exceptions.HTTPError:
                    print(f"HTTPError: {alex.works_idmap[1363]} has reference {ref_alexid} which can't be fetched by OpenAlex!")
                    continue
                ref_year = ref_work["publication_year"]
                ref_year_map[ref_alexid] = ref_year
            citation_ages.append(year-ref_year)
        return citation_ages, _id, save_path, ref_year_map, ref_year_savepath

def main(multiprocess=False):
    years = [1950]
    prefix = "openaccess_en_geq2author_hascitation_year"
    ref_year_savepath = "./data/ref_year.pkl"
    for year in years:
        alexpath = f"./data/{prefix}{year}.pkl"
        alex = Alex(load_path=alexpath)
        citation_counts = 0
        for _id in alex.works_info.keys():
            citation_counts += len(alex.works_info[_id]['citations'])
        print(f"Total {citation_counts} citations in year {year}!")
        citation_ages = dict()  # {paper: [citation_age, ...]}
        path = f"./result/citation_age/{prefix}{year}.txt"
        if os.path.exists(path) is False:
            fetched_id = set()
        else:
            fetched_id = set()
            with open(path) as f:
                for line in f.readlines():
                    fetched_id.add(int(line.strip().split(" ")[0]))
        # if multiprocess:
        #     pool = Pool(16)
        #     for _id in alex.works_info.keys():
        #         if _id in fetched_id:
        #             print(f"{_id} in {year} fetched!")
        #             continue
        #         pool.apply_async(fetch_citation_age, args=(alex, _id, year, path, ref_year_savepath, ), callback=write_results, error_callback=print_error)
        #     pool.close()
        #     pool.join()
        # else:
            # for _id in tqdm(alex.works_info.keys(), desc=f"Calc Citation Ages for work in {year}"):
        for _id in alex.works_info.keys():
            if _id in fetched_id:
                print(f"{_id} in {year} fetched!")
                continue
            result = fetch_citation_age(alex, _id, year, path, ref_year_savepath, multiprocess)
            write_results(result)

if __name__ == "__main__":
    main(multiprocess=False)