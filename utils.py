import pandas as pd
import arxiv
import requests
from pinecone import Pinecone, ServerlessSpec
from config import ZOTERO_API_KEY, ZOTERO_LIBRARY_ID, ZOTERO_TAG, PINECONE_API_KEY, INDEX_NAME, NAMESPACE_NAME, HF_API_KEY, ARXIV_CATEGORY_NAME, ARXIV_COMMENT_QUERY
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir) 

def get_zotero_ids(api_key, library_id, tag):

    base_url = 'https://api.zotero.org'
    suffix = '/users/'+ library_id +'/items?tag='+ tag

    header = {'Authorization': 'Bearer '+ api_key}
    request = requests.get(base_url + suffix, headers= header)
    
    return [data['data']['archiveID'].replace('arXiv:', '') for data in request.json()]

def get_arxiv_papers(ids = None, category = None, comment = None):

    logging.getLogger('arxiv').setLevel(logging.WARNING)

    client = arxiv.Client()

    if category is None:
        search = arxiv.Search(
            id_list= ids,
            max_results= len(ids),
        )
    else :
        if comment is None:
            custom_query = f'cat:{category}'
        else:
            custom_query = f'cat:{category} AND co:{comment}'

        search = arxiv.Search(
            query = custom_query,
            max_results= 15,
            sort_by= arxiv.SortCriterion.SubmittedDate
        )
    if ids is None and category is None:
        raise ValueError('not a valid query')

    df = pd.DataFrame({'Title': [result.title for result in client.results(search)],
                'Abstract': [result.summary.replace('\n', ' ') for result in client.results(search)],
                'Date': [result.published.date().strftime('%Y-%m-%d') for result in client.results(search)],
                'id': [result.entry_id for result in client.results(search)]})
    
    if ids:
        df.to_csv('arxiv-scrape.csv', index = False)
    return df

def get_hf_embeddings(api_key, df):

    title_abs = [title + '[SEP]' + abstract for title,abstract in zip(df['Title'], df['Abstract'])]

    API_URL = "https://api-inference.huggingface.co/models/malteos/scincl"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": False})
    print(str(response.status_code) + 'This part needs an update, causing KeyError 0')
    if response.status_code == 503:
        response = requests.post(API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": True})

    embeddings = response.json()

    return embeddings, len(embeddings[0])


def upload_to_pinecone(api_key, index, namespace, embeddings, dim, df):
    input = [{'id': df['id'][i], 'values': embeddings[i]} for i in range(len(embeddings))]

    pc = Pinecone(api_key = api_key)
    if index in pc.list_indexes().names():
        while True:
            logging.warning(f'Index name : {index} already exists.')
            return f'Index name : {index} already exists'
                
    pc.create_index(
        name=index,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

    index = pc.Index(index)
    return index.upsert(vectors=input, namespace=namespace)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  
    logging.basicConfig(filename= 'logs/logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('arxiv').setLevel(logging.WARNING)
    logging.info("Project Initialization Script Started (Serverless)")
    
    ids = get_zotero_ids(ZOTERO_API_KEY, ZOTERO_LIBRARY_ID, ZOTERO_TAG)
    print(ids)

    df = get_arxiv_papers(ids = ids)

    embeddings, dim = get_hf_embeddings(HF_API_KEY, df)

    feedback = upload_to_pinecone(PINECONE_API_KEY, INDEX_NAME, NAMESPACE_NAME, embeddings, dim, df)

    logging.info(feedback)
    if feedback is dict:
        return f"Retrieved {len(ids)} papers from Zotero. Successfully upserted {feedback['upserted_count']} embeddings in {NAMESPACE_NAME} namespace."
    else :
        return feedback

def get_new_papers(df):
    df_main = pd.read_csv('arxiv-scrape.csv')
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    union_df = df.merge(df_main, how='left', indicator=True)
    df = union_df[union_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    if df.empty:
        return 'No New Papers Found'
    else:
        df_main = pd.concat([df_main, df], ignore_index= True)
        df_main.drop_duplicates(inplace= True)
        df_main.to_csv('arxiv-scrape.csv', index = False)
        return df

def recommend_papers(api_key, index, namespace, embeddings, df, threshold):

    pc = Pinecone(api_key = api_key)
    if index in pc.list_indexes().names():
        index = pc.Index(index)
    else:
        raise ValueError(f"{index} doesnt exist. Project isnt initialized properly")
    
    results = []
    score_threshold = threshold
    for i,embedding in enumerate(embeddings):
        query = embedding
        result = index.query(namespace=namespace,vector=query,top_k=3,include_values=False)
        sum_score = sum(match['score'] for match in result['matches'])
        if sum_score > score_threshold:
            results.append(f"Paper-URL : {df['id'][i]} with score: {sum_score}")

    if results:
        return '\n'.join(results)
    else:
        return 'No Interesting Paper'



def recs(threshold):
    logging.info("Weekly Script Started (Serverless)")

    df = get_arxiv_papers(category= ARXIV_CATEGORY_NAME, comment= ARXIV_COMMENT_QUERY)

    df = get_new_papers(df)

    if not isinstance(df, pd.DataFrame):
        return df

    embeddings, _ = get_hf_embeddings(HF_API_KEY, df)

    results = recommend_papers(PINECONE_API_KEY, INDEX_NAME, NAMESPACE_NAME, embeddings, df, threshold)

    return results





if __name__ == '__main__':
    choice = int(input("1. Initialize\n2. Recommend Papers\n"))
    if choice == 1:
        print(main())
    elif choice == 2:
        threshold = float(input('Enter Similarity Threshold'))
        print(recs(threshold))
    else:
        raise ValueError('Invalid Input')
