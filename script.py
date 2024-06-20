'''
Weekly Scripts to scrape the arxiv
'''
import numpy as np
import pandas as pd
from datetime import date
import arxiv
import requests
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from config import pinecone_api_key



