import streamlit as st
from st_files_connection import FilesConnection
from json import JSONDecodeError

from utils import get_files

import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Toxicity Demo", page_icon=":exclamation:")
st.markdown("# :exclamation: Toxicity Demo")
st.write(
    """
    
    The data sets can be found at [HuggingFace](https://huggingface.co/datasets).
    
    You can check out the original [HF data set explorer app repo](https://github.com/streamlit/files-connection/tree/main/hf-example)..
    """
)

conn = st.experimental_connection('hf', type=FilesConnection)

with st.expander('Find dataset examples'):
    if 'dataset' not in st.session_state:
        st.session_state.dataset = "EleutherAI/lambada_openai"

    def set_dataset():
        st.session_state.dataset = st.session_state._dataset

    dataset_examples = [
        "EleutherAI/lambada_openai",
        "argilla/news-summary",
        "stanfordnlp/SHP",
        "HuggingFaceM4/tmp-pmd-synthetic-testing",
    ]
    st.selectbox("Examples", dataset_examples, key="_dataset", on_change=set_dataset)
    """
        You can also search for data sets on [HuggingFace Hub](https://huggingface.co/datasets).
        This repo supports data sets with data stored in json, jsonl, csv or parquet format.
    """

# Enter a dataset and retrieve a list of data files
#dataset_name = st.text_input("Enter your dataset of interest", key='dataset')
#dataset = Path('datasets', dataset_name)
dataset = Path('datasets',st.session_state.dataset)
dataset = dataset.as_posix()
file_names = get_files(conn, dataset)

if not file_names:
    st.warning(
        "No compatible data files found. This app only supports datasets stored in csv, json[l] or parquet format.")
    st.stop()

# Select a data file and row count to retrieve
file_selection = st.selectbox("Pick a data file", file_names)
datafile = Path(dataset, file_selection)
nrows = st.slider("Rows to retrieve", value=50)

"## Dataset Preview"
#kwargs = dict(nrows=nrows, ttl=3600)
kwargs = dict(ttl=3600)

# parquet doesn't neatly support nrows
# could be fixed with something like this:
# https://stackoverflow.com/a/69888274/20530083
#if datafile.suffix in ('.parquet', '.json'):
#    del (kwargs['nrows'])

try:
    df = conn.read(datafile.as_posix(), **kwargs)
except JSONDecodeError as e:
    # often times because a .json file is really .jsonl
    try:
        df = conn.read(datafile.as_posix(), input_format='jsonl', nrows=nrows, **kwargs)
    except:
        raise e

st.dataframe(df.head(nrows), use_container_width=True)

def clean_data(data):
    clean_data = data.dropna()
    return clean_data

def process_text(input_text):
    lowercase = input_text.lower() # lowercase
    replaced = re.sub(r'[^\w\s]', ' ', lowercase)  # remove any special characters
    return replaced

def find_match_count(text: str, pattern: str) -> int:
    return len(re.findall(r'\b' + pattern + r'\b', text))

clean_df = clean_data(df)
text_col_name = st.text_input('Name of the column with text is', 'text')
processed_df = clean_df[text_col_name].apply(lambda x: process_text(x))




# from cleanlab_studio import Studio

# API_KEY = os.getenv("CLEANLAB_API_KEY")

# # initialize studio object
# studio = Studio(API_KEY)

# dataset_id = studio.upload_dataset(clean_df, dataset_name="text-toxicity-quickstart")
# st.write(f"Dataset ID: {dataset_id}")

# project_id = studio.create_project(
#     dataset_id=dataset_id,
#     project_name="text-toxicity-project",
#     modality="text",
#     task_type="multi-class",
#     model_type="regular",
#     label_column="label",
#     text_column="text",
# )
# print(f"Project successfully created and training has begun! project_id: {project_id}")