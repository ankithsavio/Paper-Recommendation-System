'''
Weekly Script to scrape the arxiv.
Guide to creating a config file is in init-script.
'''
import pandas as pd
import arxiv
from pinecone import Pinecone
from config import PINECONE_API_KEY, INDEX_NAME, NAMESPACE_NAME, CHECKPOINT_PATH, ARXIV_CATEGORY_NAME, ARXIV_COMMENT_QUERY, HF_API_KEY
import logging
import requests
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

logging.basicConfig(filename= 'logs/logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('arxiv').setLevel(logging.WARNING)
logging.info("Weekly Script Started (Serverless)")

client = arxiv.Client()
category = ARXIV_CATEGORY_NAME
comment = ARXIV_COMMENT_QUERY
custom_query = f'cat:{category} AND co:{comment}'

search = arxiv.Search(
    query = custom_query,
    max_results= 15,
    sort_by= arxiv.SortCriterion.SubmittedDate
)

results = list(client.results(search))
if not results:
    logging.warning("No results found for the given query. Check the README file for the necessary info.")
    exit()

df = pd.DataFrame({'Title': [result.title for result in client.results(search)],
              'Abstract': [result.summary.replace('\n', ' ') for result in client.results(search)],
              'Date': [result.published.date().strftime('%Y-%m-%d') for result in client.results(search)],
              'id': [result.entry_id for result in client.results(search)]})

try:
    df_main = pd.read_csv('arxiv-scrape.csv')
except FileNotFoundError:
    logging.warning("CSV file not found, Project isnt initialized.")
    exit()

df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
union_df = df.merge(df_main, how='left', indicator=True)
df = union_df[union_df['_merge'] == 'left_only'].drop(columns=['_merge'])

if df.empty:
    logging.info("No new papers found.")
else:
    df_main = pd.concat([df_main, df], ignore_index= True)
    df_main.drop_duplicates(inplace= True)
    df_main.to_csv('arxiv-scrape.csv', index = False)

    title_abs = [title + '[SEP]' + abstract for title,abstract in zip(df['Title'], df['Abstract'])]

    API_URL = "https://api-inference.huggingface.co/models/malteos/scincl"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": False})
    if response.status_code == 502:
        response = requests.post(API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": True})

    embeddings = response.json()

    pc = Pinecone(api_key = PINECONE_API_KEY)
    if INDEX_NAME in pc.list_indexes().names():
        index = pc.Index(INDEX_NAME)
    else:
        logging.error(f"{INDEX_NAME} doesnt exist. Project isnt initialized properly")
        exit()

    results = []
    score_threshold = 2.61
    for i,embedding in enumerate(embeddings):
        query = embedding[i]
        result = index.query(namespace=NAMESPACE_NAME,vector=query,top_k=3,include_values=False)
        sum_score = sum(match['score'] for match in result['matches'])
        if sum_score > score_threshold:
            results.append({'id':df['id'][i], 'total_score': sum_score})


    if results:
        logging.info(results)
    else:
        logging.info("No Interesting Papers")

logging.info("Weekly Paper Recommendation Script Completed.")

