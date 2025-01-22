"""
Initialization script for the project.
For the current state of the project, it is desirable to keep the syntax as is.
"""

import pandas as pd
import arxiv
import requests
from pinecone import Pinecone, ServerlessSpec
import logging
import os
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs("logs/", exist_ok=True)
logging.basicConfig(
    filename="logs/logfile.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.info("Project Initialization Script Started (Serverless)")

library_type = "user"
base_url = "https://api.zotero.org"
suffix = (
    "/users/" + os.getenv("ZOTERO_LIBRARY_ID") + "/items?tag=" + os.getenv("ZOTERO_TAG")
)

header = {"Authorization": "Bearer " + os.getenv("ZOTERO_API_KEY")}
request = requests.get(base_url + suffix, headers=header)

ids = [data["data"]["archiveID"].replace("arXiv:", "") for data in request.json()]

client = arxiv.Client()
search = arxiv.Search(
    id_list=ids,
    max_results=len(ids),
)

df = pd.DataFrame(
    {
        "Title": [result.title for result in client.results(search)],
        "Abstract": [
            result.summary.replace("\n", " ") for result in client.results(search)
        ],
        "Date": [
            result.published.date().strftime("%Y-%m-%d")
            for result in client.results(search)
        ],
        "id": [
            result.entry_id.replace("http://arxiv.org/abs/", "")
            for result in client.results(search)
        ],
    }
)

df.to_csv("arxiv-scrape.csv", index=False)
title_abs = [
    title + "[SEP]" + abstract for title, abstract in zip(df["Title"], df["Abstract"])
]

API_URL = "https://api-inference.huggingface.co/models/malteos/scincl"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

response = requests.post(
    API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": False}
)
if response.status_code == 503:
    response = asyncio.run(
        asyncio.to_thread(
            requests.post,
            API_URL,
            headers=headers,
            json={"inputs": title_abs, "wait_for_model": True},
        )
    )
    # response = requests.post(
    #     API_URL, headers=headers, json={"inputs": title_abs, "wait_for_model": True}
    # )

output = response.json()
embedding_vector = [
    {"id": df["id"][i], "values": output[i]} for i in range(len(output))
]

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if os.getenv("INDEX_NAME") in pc.list_indexes().names():
    while True:
        logging.warning(f"Index name : {os.getenv('INDEX_NAME')} already exists.")
        response = input(
            f"Index name : {os.getenv('INDEX_NAME')} already exists. Set a different name in your config file OR you can proceed to re-initialize the project.\n"
            + "Do you want to proceed? [y/n]:"
        )
        if response.lower() == "y":
            pc.delete_index(os.getenv("INDEX_NAME"))
            break
        elif response.lower() == "n":
            logging.info("Project Terminated.")
            exit()
        else:
            print("Invalid input. Use [y/n].")

pc.create_index(
    name=os.getenv("INDEX_NAME"),
    dimension=len(output[0]),
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(os.getenv("INDEX_NAME"))
feedback = index.upsert(vectors=embedding_vector, namespace=os.getenv("NAMESPACE_NAME"))

logging.info(feedback)
logging.info(
    f"Retrieved {len(ids)} papers from Zotero. Successfully upserted {feedback['upserted_count']} embeddings in {os.getenv('NAMESPACE_NAME')} namespace."
)
