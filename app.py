import logging
import os
import gradio as gr
import pandas as pd
from pinecone import Pinecone
from utils import (
    get_zotero_ids,
    get_arxiv_papers,
    get_hf_embeddings,
    upload_to_pinecone,
    get_new_papers,
    recommend_papers,
)
from dotenv import load_dotenv

load_dotenv(".env")
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
NAMESPACE_NAME = os.getenv("NAMESPACE_NAME")

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def category_radio(cat):
    if cat == "Computer Vision and Pattern Recognition":
        return "cs.CV"
    elif cat == "Computation and Language":
        return "cs.CL"
    elif cat == "Artificial Intelligence":
        return "cs.AI"
    elif cat == "Robotics":
        return "cs.RO"


def comment_radio(com):
    if com == "None":
        return None
    else:
        return com


def reset_project():
    file_path = "arxiv-scrape.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(
            f"{file_path} has been deleted. Delete reset_project() if you want to persist recommended papers."
        )

    api_key = os.getenv("PINECONE_API_KEY")
    index = os.getenv("INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    if index in pc.list_indexes().names():
        pc.delete_index(index)
        logging.info(
            f"{index} index has been deleted from the vectordb. Delete reset_project() if you want to persist recommended papers."
        )
    return f"{file_path} has been deleted.<br />{index} index has been deleted from the vectordb.<br />"


def reset_csv():
    file_path = "arxiv-scrape.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(
            f"{file_path} has been deleted. Delete reset_project() if you want to persist recommended papers."
        )


with gr.Blocks() as demo:

    zotero_api_key = gr.Textbox(
        label="Zotero API Key", type="password", value=os.getenv("ZOTERO_API_KEY")
    )

    zotero_library_id = gr.Textbox(
        label="Zotero Library ID", value=os.getenv("ZOTERO_LIBRARY_ID")
    )

    zotero_tag = gr.Textbox(label="Zotero Tag", value=os.getenv("ZOTERO_TAG"))

    arxiv_category_name = gr.State([])
    radio_arxiv_category_name = gr.Radio(
        [
            "Computer Vision and Pattern Recognition",
            "Computation and Language",
            "Artificial Intelligence",
            "Robotics",
        ],
        value=["Computer Vision and Pattern Recognition"],
        label="ArXiv Category Query",
    )
    radio_arxiv_category_name.change(
        fn=category_radio, inputs=radio_arxiv_category_name, outputs=arxiv_category_name
    )

    arxiv_comment_query = gr.State([])
    radio_arxiv_comment_query = gr.Radio(
        ["CVPR", "ACL", "TACL", "JAIR", "IJRR", "None"],
        value=["CVPR"],
        label="ArXiv Comment Query",
    )
    radio_arxiv_comment_query.change(
        fn=comment_radio, inputs=radio_arxiv_comment_query, outputs=arxiv_comment_query
    )

    threshold = gr.Slider(
        minimum=0.70, maximum=0.99, value=0.80, label="Similarity Score Threshold"
    )

    init_output = gr.Textbox(label="Project Initialization Result")

    rec_output = gr.Markdown(label="Recommended Papers")

    reset_output = gr.Markdown(label="Reset Declaration")

    init_btn = gr.Button("Initialize")

    rec_btn = gr.Button("Recommend")

    reset_btn = gr.Button("Reset")

    reset_btn.click(fn=reset_project, inputs=[], outputs=[reset_output])

    @init_btn.click(
        inputs=[zotero_api_key, zotero_library_id, zotero_tag], outputs=[init_output]
    )
    def init(
        zotero_api_key,
        zotero_library_id,
        zotero_tag,
        hf_api_key=HF_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        namespace_name=NAMESPACE_NAME,
    ):

        logging.basicConfig(
            filename="logfile.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Project Initialization Script Started (Serverless)")

        ids = get_zotero_ids(zotero_api_key, zotero_library_id, zotero_tag)

        df = get_arxiv_papers(ids)

        embeddings, dim = get_hf_embeddings(hf_api_key, df)

        feedback = upload_to_pinecone(
            pinecone_api_key, index_name, namespace_name, embeddings, dim, df
        )

        logging.info(feedback)
        if isinstance(feedback, dict):
            return f"Retrieved {len(ids)} papers from Zotero. Successfully upserted {feedback['upserted_count']} embeddings in {namespace_name} namespace."
        else:
            return feedback

    @rec_btn.click(
        inputs=[arxiv_category_name, arxiv_comment_query, threshold],
        outputs=[rec_output],
    )
    def recs(
        arxiv_category_name,
        arxiv_comment_query,
        threshold,
        hf_api_key=HF_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        namespace_name=NAMESPACE_NAME,
    ):
        logging.info("Weekly Script Started (Serverless)")

        df = get_arxiv_papers(category=arxiv_category_name, comment=arxiv_comment_query)

        df = get_new_papers(df)

        if not isinstance(df, pd.DataFrame):
            return df

        embeddings, _ = get_hf_embeddings(hf_api_key, df)

        results = recommend_papers(
            pinecone_api_key, index_name, namespace_name, embeddings, df, threshold * 3
        )

        return results


demo.launch(share=True)
