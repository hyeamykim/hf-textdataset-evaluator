"""Streamlit app for Presidio."""
import logging
import os
# import traceback
from pathlib import Path

# import dotenv
# from dotenv import load_dotenv, find_dotenv
import pandas as pd
from json import JSONDecodeError
import re
# import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# import seaborn as sns
# from collections import Counter
# import nltk
# from nltk.corpus import stopwords

import streamlit as st
from st_files_connection import FilesConnection
# from annotated_text import annotated_text
from streamlit_tags import st_tags
# from presidio_structured import StructuredEngine, PandasAnalysisBuilder, StructuredAnalysis, CsvReader, PandasDataProcessor

from utils import get_files
from presidio_helpers import (
    get_supported_entities,
    analyze,
    # anonymize,
    # annotate,
    analyzer_engine,
    batch_analyzer_engine,
    batch_analyze,
    # batch_anonymize
)

st.set_page_config(
    page_title="Presidio demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
    page_icon=":id:",
)

# dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")

allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# Main panel
st.header(
    """
    :id: PII De-Identification with [Microsoft Presidio](https://microsoft.github.io/presidio/)
    """
)

with st.expander("About this demo", expanded=False):
    st.info(
        """Presidio is an open source customizable framework for PII detection and de-identification.
        \n\n[Code](https://aka.ms/presidio) | 
        [Tutorial](https://microsoft.github.io/presidio/tutorial/) | 
        [Installation](https://microsoft.github.io/presidio/installation/) | 
        [FAQ](https://microsoft.github.io/presidio/faq/) |
        [Feedback](https://forms.office.com/r/9ufyYjfDaY) |"""
    )

    st.info(
        """
    Use this demo to:
    - Experiment with different off-the-shelf models and NLP packages.
    - Explore the different de-identification options, including redaction, masking, encryption and more.
    - Generate synthetic text with Microsoft Presidio and OpenAI.
    - Configure allow and deny lists.
    
    This demo website shows some of Presidio's capabilities.
    [Visit our website](https://microsoft.github.io/presidio) for more info,
    samples and deployment options.    
    """
    )

    st.markdown(
        "[![Pypi Downloads](https://img.shields.io/pypi/dm/presidio-analyzer.svg)](https://img.shields.io/pypi/dm/presidio-analyzer.svg)"  # noqa
        "[![MIT license](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)"
        "![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/presidio?style=social)"
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
# dataset_name = st.text_input("Enter your dataset of interest", key='dataset')
#dataset = Path('datasets', dataset_name)
dataset = Path('datasets', st.session_state.dataset)
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
processed_df = clean_df.copy()
processed_df[text_col_name] = clean_df[text_col_name].apply(lambda x: process_text(x))

model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    Presidio supports multiple NER packages off-the-shelf, such as spaCy and Huggingface.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    "spaCy/en_core_web_lg",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "Other",
]
if not allow_other_models:
    model_list.pop()
# Select model
st_model = st.selectbox(
    "NER model package",
    model_list,
    index=2,
    help=model_help_text,
)

# Extract model package.
st_model_package = st_model.split("/")[0]

# Remove package prefix (if needed)
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.selectbox(
        "NER model OSS package", options=["spaCy", "HuggingFace"]
    )
    st_model = st.text_input(f"NER model name", value="")

st.warning("Note: Models might take some time to download. ")

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
logger.debug(f"analyzer_params: {analyzer_params}")

# st_operator = st.selectbox(
#     "De-identification approach",
#     ["redact", "replace", "highlight", "mask", "hash", "encrypt"],
#     index=1,
#     help="""
#     Select which manipulation to the text is requested after PII has been identified.\n
#     - Redact: Completely remove the PII text\n
#     - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
#     - Highlight: Shows the original text with PII highlighted in colors\n
#     - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
#     - Hash: Replaces with the hash of the PII string\n
#     - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
#          """,
# )
# st_mask_char = "*"
# st_number_of_chars = 15
# st_encrypt_key = "WmZq4t7w!z%C&F)J"

# logger.debug(f"st_operator: {st_operator}")

# if st_operator == "mask":
#     st_number_of_chars = st.number_input(
#         "number of chars", value=st_number_of_chars, min_value=0, max_value=100
#     )
#     st_mask_char = st.text_input(
#         "Mask character", value=st_mask_char, max_chars=1
#     )
# elif st_operator == "encrypt":
#     st_encrypt_key = st.text_input("AES key", value=st_encrypt_key)

st_threshold = st.slider(
    label="Acceptance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Define the threshold for accepting a detection as PII. See more here: ",
)

st_return_decision_process = st.checkbox(
    "Add analysis explanations to findings",
    value=False,
    help="Add the decision process to the output table. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/decision_process/",
)

# Allow and deny lists
st_deny_allow_expander = st.expander(
    "Allowlists and denylists",
    expanded=False,
)

with st_deny_allow_expander:
    st_allow_list = st_tags(
        label="Add words to the allowlist", text="Enter word and press enter."
    )
    st.caption(
        "Allowlists contain words that are not considered PII, but are detected as such."
    )

    st_deny_list = st_tags(
        label="Add words to the denylist", text="Enter word and press enter."
    )
    st.caption(
        "Denylists contain words that are considered PII, but are not detected as such."
    )

st_entities_expander = st.expander("Choose entities to look for")
st_entities = st_entities_expander.multiselect(
    label="Which entities to look for?",
    options=get_supported_entities(*analyzer_params),
    default=list(get_supported_entities(*analyzer_params)),
    help="Limit the list of PII entities detected. "
    "This list is dynamic and based on the NER model and registered recognizers. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
)

# analyzer_load_state = st.info("Starting Presidio analyzer...")

# analyzer_load_state.empty()

# # Read default text
# with open("demo_text.txt") as f:
#     demo_text = f.readlines()

# # Create two columns for before and after
# col1, col2 = st.columns(2)

# # Before:
# col1.subheader("Input")
# st_text = col1.text_area(
#     label="Enter text", value="".join(demo_text), height=400, key="text_input"
# )

# try:
#     # Choose entities
#     st_entities_expander = st.expander("Choose entities to look for")
#     st_entities = st_entities_expander.multiselect(
#         label="Which entities to look for?",
#         options=get_supported_entities(*analyzer_params),
#         default=list(get_supported_entities(*analyzer_params)),
#         help="Limit the list of PII entities detected. "
#         "This list is dynamic and based on the NER model and registered recognizers. "
#         "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
#     )

#     # Before
#     analyzer_load_state = st.info("Starting Presidio analyzer...")
#     analyzer = analyzer_engine(*analyzer_params)
#     analyzer_load_state.empty()

#     st_analyze_results = analyze(
#         *analyzer_params,
#         text=st_text,
#         entities=st_entities,
#         language="en",
#         score_threshold=st_threshold,
#         return_decision_process=st_return_decision_process,
#         allow_list=st_allow_list,
#         deny_list=st_deny_list,
#     )

#     # After
#     if st_operator not in ("highlight"):
#         with col2:
#             st.subheader(f"Output")
#             st_anonymize_results = anonymize(
#                 text=st_text,
#                 operator=st_operator,
#                 mask_char=st_mask_char,
#                 number_of_chars=st_number_of_chars,
#                 encrypt_key=st_encrypt_key,
#                 analyze_results=st_analyze_results,
#             )
#             st.text_area(
#                 label="De-identified", value=st_anonymize_results.text, height=400
#             )
#     else:
#         st.subheader("Highlighted")
#         annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
#         # annotated_tokens
#         annotated_text(*annotated_tokens)

#     # table result
#     st.subheader(
#         "Findings"
#         if not st_return_decision_process
#         else "Findings with decision factors"
#     )
#     if st_analyze_results:
#         df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
#         df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

#         df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
#             {
#                 "entity_type": "Entity type",
#                 "text": "Text",
#                 "start": "Start",
#                 "end": "End",
#                 "score": "Confidence",
#             },
#             axis=1,
#         )
#         df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
#         if st_return_decision_process:
#             analysis_explanation_df = pd.DataFrame.from_records(
#                 [r.analysis_explanation.to_dict() for r in st_analyze_results]
#             )
#             df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
#         st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
#     else:
#         st.text("No findings")

# except Exception as e:
#     print(e)
#     traceback.print_exc()
#     st.error(e)

# sample size calculator
population = clean_df.shape[0]
z_score = 1.96 # for 95% confidence level
st_dev = 0.5
margin_of_error = 0.05
numerator = (z_score**2 * st_dev * (1 - st_dev)) / margin_of_error**2
denominator = 1 + ((z_score**2 * st_dev * (1 - st_dev)) / (margin_of_error**2 * population))
sample_size = round(numerator/denominator)

sample_df = clean_df.sample(n=sample_size, random_state=42)
# st.dataframe(sample_df.head())

sample_indices = sample_df.index

# text_list = clean_df[text_col_name].values.tolist()
text_list = sample_df[text_col_name].values.tolist()

df_dict = sample_df.to_dict(orient="list")

batch_analyzer_load_state = st.info("Starting Presidio analyzer...")
batch_analyzer = batch_analyzer_engine(*analyzer_params)
batch_analyzer_load_state.empty()

st_batch_analyze_results = batch_analyze(
        *analyzer_params,
        texts=text_list,
        entities=st_entities,
        language="en",
        score_threshold=st_threshold,
        return_decision_process=st_return_decision_process,
        allow_list=st_allow_list,
        deny_list=st_deny_list,
    )

# st.write(st_batch_analyze_results[0:5])
# 0:"type: ORGANIZATION, start: 275, end: 281, score: 0.8686715960502625"

len_list = [len(item) for item in st_batch_analyze_results]
index_list = [[i]*num for i, num in zip(sample_indices, len_list)]
index_list = [x for xs in index_list for x in xs]
# st.write(len(index_list)) # 865

# https://realpython.com/python-flatten-list/#flattening-a-list-using-standard-library-and-built-in-tools
flatten_results = [item.to_dict() for r in st_batch_analyze_results for item in r]
# st.write(len(flatten_results)) # 12364 # 865
# st.write(flatten_results[0:5])

# https://www.geeksforgeeks.org/python-convert-key-value-string-to-dictionary/
# dict_results = { 
#     item.split(":")[0]: item.split(":")[1]
#     for item in [items.split(",") for items in flatten_results]
# }

# dict_results = { 
#     item.split(":")[0]: item.split(":")[1]
#     for item in [items.split(",") for items in st_batch_analyze_results]
# }

# https://stackoverflow.com/questions/4260280/if-else-in-a-list-comprehension
# [f(x) if x is not None else '' for x in xs]

# dict_results = { 
#     item.split(":")[0]: item.split(":")[1]
#     for item in [items.split(",") if isintance(items, string) for items in st_batch_analyze_results]
# }
# st.write(next(iter(dict_results.items())))
# batch_df = pd.DataFrame(dict_results)


# https://stackoverflow.com/questions/14071038/add-an-element-in-each-dictionary-of-a-list-list-comprehension
# elements = ['value'] * len(myList)
# result = map(lambda item: dict(item[0], elem=item[1]),zip(myList,elements))
# results = map(lambda item: dict(item[0], index=item[1]), zip(flatten_results,sample_indices))
# st.write(results)

# https://www.geeksforgeeks.org/python-add-custom-values-key-in-list-of-dictionaries/
K = 'index_num'
# test_list = list(map(lambda x, y: {**x, K: y}, test_list, append_list))
flatten_results = list(map(lambda x, y:{**x, K:y}, flatten_results, index_list))
st.write(flatten_results[0:5])

flatten_df = pd.DataFrame(flatten_results)
st.dataframe(flatten_df.head())

flatten_df = flatten_df.set_index('index_num')
results_df = pd.merge(flatten_df, sample_df, left_index=True, right_index=True)
results_df['identity'] = results_df.apply(
    lambda row: row['text'][row['start']:row['end']] if 0 <= row['start'] < len(row['text']) and 0 <= row['end'] <= len(row['text']) else '', 
    axis=1
)
st.dataframe(results_df.head())

count_df = results_df['entity_type'].value_counts()
st.dataframe(count_df)

name_df = results_df['identity'].value_counts()
st.dataframe(name_df)

fig, ax = plt.subplots(figsize=(6, 4))
count_df.plot.barh(ax=ax)
ax.set_title("Most Frequently detected PII entities in the Dataset")
st.pyplot(fig)

# bins = 10
# fig, ax = plt.subplots(figsize=(5,4))
# ax.hist(results_df['identity'], bins, alpha=0.5, label='identity')
# ax.legend(loc='upper right')
# ax.set_title('Histogram of detected PII words in the Dataset')
# st.pyplot(fig)

# batch_df = pd.DataFrame.from_records(st_batch_analyze_results)
# batch_df.assign(combined=batch_df.agg(list, axis=1))
# batch_df['combined'] = batch_df[batch_df.columns[0:]].apply(
#     lambda x: x.dropna().to_dict(),
#     axis=1
# )
# dict_df = pd.DataFrame.from_dict(batch_df['combined'])
# st.dataframe(dict_df.head())


# batch_df.drop(columns=['combined'], inplace=True)
# batch_df.stack(future_stack=True).droplevel(-1).reset_index(name='analyzer results')
# st.dataframe(batch_df.head())

# batch_df = pd.DataFrame.from_records(st_batch_analyze_results)
# out_df = batch_df.stack().droplevel(-1).reset_index(name='analyzer results')

# batch_df = pd.DataFrame.from_records([r.to_dict() for r in st_batch_analyze_results])
# batch_df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

# results_df = processed_df.copy()
# results_df['analyer results'] = st_batch_analyzer_results
# results_df = pd.DataFrame({'analyzer results': st_batch_analyze_results})
# st.dataframe(results_df.head(), use_container_width=True)

# results_df = pd.DataFrame({'analyzer results': flatten_results})
# st.dataframe(results_df.head(), use_container_width=True)


