'''
Weekly Script to scrape the arxiv.
Guide to creating a config file is in init-script.
'''
import pandas as pd
import arxiv
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from config import pinecone_api_key, arxiv_category_name, arxiv_comment_query, checkpoint_path, index_name, namespace_name
import torch
import math

client = arxiv.Client()

category = arxiv_category_name
comment = arxiv_comment_query
custom_query = f'cat:{category} AND co:{comment}'

search = arxiv.Search(
    query = custom_query,
    max_results= 15,
    sort_by= arxiv.SortCriterion.SubmittedDate
)

df = pd.DataFrame({'Title': [result.title for result in client.results(search)],
              'Abstract': [result.summary.replace('\n', ' ') for result in client.results(search)],
              'Date': [result.published.date().strftime('%Y-%m-%d') for result in client.results(search)],
              'id': [result.entry_id for result in client.results(search)]})
print(df.size)
df_main = pd.read_csv('arxiv-scrape.csv')

union_df = df.merge(df_main, how='left', indicator=True)
df = union_df[union_df['_merge'] == 'left_only'].drop(columns=['_merge'])
print(df.size)
df_main = pd.concat([df_main, df], ignore_index= True)
df_main.drop_duplicates(inplace= True)
df_main.to_csv('arxiv-scrape.csv')

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModel.from_pretrained(checkpoint_path)

title_abs = [title + tokenizer.sep_token + abstract for title,abstract in zip(df['Title'], df['Abstract'])]

embeddings = torch.empty((0, 768))
batch_size = 15
for i in range(math.ceil(len(title_abs)/batch_size)):
    inputs = tokenizer(title_abs[i * batch_size:(i + 1) * batch_size], padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    embeddings = torch.cat((embeddings, result.last_hidden_state[:, 0, :]), dim = 0)

pc = Pinecone(api_key = pinecone_api_key)
if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
else:
    print(f"{index_name} doesnt exist. Run init-script first.")
    exit()

results = []
score_threshold = 2.2
for i,embedding in enumerate(embeddings):
    query = embedding.detach().numpy().tolist()
    result = index.query(namespace=namespace_name,vector=query,top_k=3,include_values=False)
    sum_score = 0
    for match in result['matches']:
        sum_score += match['score'] 
    if sum_score > score_threshold:
        results.append({'id':df['id'][i], 'total_score': sum_score})


for result in results:
    print(f"Possible interesting paper : {result['id']} \nwith Average Score = {result['total_score']/3}")