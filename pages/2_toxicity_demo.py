import streamlit as st
from st_files_connection import FilesConnection
from json import JSONDecodeError
from utils import get_files, clean_data, process_text, find_match_count
import os                                                                                                                                                                                                  
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
# from sklearn.metrics import roc_auc_score

import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

from detoxify import Detoxify

st.set_page_config(page_title="Toxicity Demo", page_icon=":exclamation:")
st.markdown("# :exclamation: Toxicity Demo")
st.write(
    """
    unbiased-toxic-roberta model trained on datsets by Unitary was used for toxicity classification.
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

clean_df = clean_data(df)
text_col_name = st.text_input('Name of the column with text is', 'text')
processed_df = clean_df[text_col_name].apply(lambda x: process_text(x))

# Calculate sample size.
population = processed_df.shape[0]
z_score = 1.96 # for 95% confidence level
st_dev = 0.5
margin_of_error = 0.05
numerator = (z_score**2 * st_dev * (1 - st_dev)) / margin_of_error**2
denominator = 1 + ((z_score**2 * st_dev * (1 - st_dev)) / (margin_of_error**2 * population))
sample_size = round(numerator/denominator)

sample_df = processed_df.sample(n=sample_size, random_state=42)
st.dataframe(sample_df.head())

text_list = sample_df.values.tolist()

# Toxicity prediction 
results = Detoxify('original').predict(text_list)
results_df = pd.DataFrame(results, index=text_list).round(5)
st.dataframe(results_df)

# Toxicity classification
bool_results_df = results_df.loc[results_df["toxicity"] >= 0.5]

n_results = bool_results_df.shape[0]
per_sample_with_toxic = round(n_results/sample_size*100,2)

st.markdown(
    f"The dataset contains :blue[{population}] records. "
    f"Using the sample size calculator, a total of :blue[{sample_size}] records were sampled. "
    f"The sampled dataset contains :blue[{n_results}] text records, :blue[{per_sample_with_toxic}]percentage of the sample dataset, "
    f"that were classified as toxic. "
)

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