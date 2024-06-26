"""
Initialization script for the project.
For the current state of the project, it is desirable to keep the syntax as is.

Pre-requisite: create an config.py
ZOTERO_LIBRARY_ID : Your Zotero userID for use in API calls
ZOTERO_API_KEY : Your private Zotero API key
ZOTERO_TAG : Desired tag from your Zotero library
PINECONE_API_KEY : Your Pinecone API key
CHECKPOINT_PATH : Path to the model checkpoints for the scincl model - Download it from Huggingface
INDEX_NAME : Your desired index name for the Vector DB - WARNING : Currently set to delete if already exists.
NAMESPACE_NAME : Your desired namspace name for the collection in the Vector DB - Project will be wokring under a single collection
"""
import pandas as pd
import arxiv
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from config import ZOTERO_API_KEY, ZOTERO_LIBRARY_ID, ZOTERO_TAG, CHECKPOINT_PATH, PINECONE_API_KEY, INDEX_NAME, NAMESPACE_NAME
import math
import warnings

library_type = 'user'

base_url = 'https://api.zotero.org'
suffix = '/users/'+ ZOTERO_LIBRARY_ID +'/items?tag='+ ZOTERO_TAG

header = {'Authorization': 'Bearer '+ ZOTERO_API_KEY}
request = requests.get(base_url + suffix, headers= header)

ids = []
for data in request.json():
    ids.append(data['data']['archiveID'].replace('arXiv:', ''))

client = arxiv.Client()
search = arxiv.Search(
    id_list= ids,
    max_results= len(ids),
)

df = pd.DataFrame({'Title': [result.title for result in client.results(search)],
              'Abstract': [result.summary.replace('\n', ' ') for result in client.results(search)],
              'Date': [result.published.date().strftime('%Y-%m-%d') for result in client.results(search)],
              'id': [result.entry_id.replace('http://arxiv.org/abs/', '') for result in client.results(search)]})

df.to_csv('arxiv-scrape.csv', index = False)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModel.from_pretrained(CHECKPOINT_PATH)

title_abs = [title + tokenizer.sep_token + abstract for title,abstract in zip(df['Title'], df['Abstract'])]

embeddings = torch.empty((0, 768))
batch_size = 15
for i in range(math.ceil(len(title_abs)/batch_size)):
    inputs = tokenizer(title_abs[i * batch_size:(i + 1) * batch_size], padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    embeddings = torch.cat((embeddings, result.last_hidden_state[:, 0, :]), dim = 0)
embedding_vector = [{'id': df['id'][i], 'values': embeddings[i]} for i in range(embeddings.shape[0])]

pc = Pinecone(api_key = PINECONE_API_KEY)
if INDEX_NAME in pc.list_indexes().names():
    while True:
        warnings.warn(f'Index name : {INDEX_NAME} already exists. Set a different name in your config file OR you can proceed to re-initialize the project.')
        response = input('Do you want to proceed? [y/n]:')
        if response.lower() == 'y':
            pc.delete_index(INDEX_NAME)
            break
        elif response.lower() == 'n':
            exit()
        else :
            print('Invalid input. Use [y/n].') 
            
pc.create_index(
    name=INDEX_NAME,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws', 
        region='us-east-1'
    ) 
) 

index = pc.Index(INDEX_NAME)
feedback = index.upsert(vectors=embedding_vector,
                        namespace=NAMESPACE_NAME)

print(f"Retrieved {len(ids)} papers from Zotero\nSuccessfully upserted {feedback['upserted_count']} embeddings in {NAMESPACE_NAME} namespace")



