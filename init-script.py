"""
Initialization script for the project.
For the current state of the project, it is desirable to keep the syntax as is.

Pre-requisite: create an config.py
zotero_library_id : Your Zotero userID for use in API calls
zotero_api_key : Your private Zotero API key
zotero_tag : Your desired tag in Zotero
pinecone_api_key : Your Pinecone API key
checkpoint_path : Path to the model checkpoints for the scincl model
index_name : Your desired index name for the Vector DB
namespace_name : Your desired namspace name for the collection in the Vector DB
"""
import pandas as pd
import arxiv
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from config import zotero_library_id, zotero_api_key, zotero_tag, pinecone_api_key, checkpoint_path, index_name, namespace_name
import math

library_type = 'user'

base_url = 'https://api.zotero.org'
suffix = '/users/'+zotero_library_id+'/items?tag='+zotero_tag

header = {'Authorization': 'Bearer '+ zotero_api_key}
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

df.to_csv('arxiv-scrape.csv')

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModel.from_pretrained(checkpoint_path)

title_abs = [title + tokenizer.sep_token + abstract for title,abstract in zip(df['Title'], df['Abstract'])]

embeddings = torch.empty((0, 768))
batch_size = 15
for i in range(math.ceil(len(title_abs)/batch_size)):
    inputs = tokenizer(title_abs[i * batch_size:(i + 1) * batch_size], padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    embeddings = torch.cat((embeddings, result.last_hidden_state[:, 0, :]), dim = 0)
embedding_vector = [{'id': df['id'][i], 'values': embeddings[i]} for i in range(embeddings.shape[0])]

pc = Pinecone(api_key = pinecone_api_key)
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

index = pc.Index(index_name)
feedback = index.upsert(vectors=embedding_vector,
                        namespace=namespace_name)

print(f"Retrieved {len(ids)} papers from Zotero\nSuccessfully upserted {feedback['upserted_count']} embeddings in {namespace_name} namespace")



