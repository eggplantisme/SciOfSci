from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from OpenAlex import *

def main():
    start = '2000-01-01'
    end = '2005-01-01'
    works = Works().filter(from_publication_date=start, to_publication_date=end, 
               language='en',
               is_oa=True,
               has_references=True,
               primary_topic={"domain":{'id': '1'}})  # Domain ID for Life Science
    print(f"From {start} to {end}: {works.count()} papers found.")
    alex = Alex(works=works)
    save_path = f"./data/Alex{start.split('-')[0]}_{end.split('-')[0]}_en_oa_hasref.pkl"
    alex.save(save_path)


if __name__ == "__main__":
    main()
    print("Done!")
