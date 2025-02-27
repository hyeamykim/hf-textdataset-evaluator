"""Streamlit app for Presidio."""
import logging
import os
from pathlib import Path # is this causing a problem?
from json import JSONDecodeError
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
# from st_files_connection import FilesConnection
from streamlit_tags import st_tags

from utils import get_files, clean_data, process_text
from presidio_helpers import (
    get_supported_entities,
    batch_analyzer_engine,
    batch_analyze,
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

logger = logging.getLogger("presidio-streamlit")
allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# Main panel
st.header(
    """
    :id: PII Detection with [Microsoft Presidio](https://microsoft.github.io/presidio/)
    """
)
st.info(
    """ 
    - This demo is built with [Microsoft Presidio App](https://huggingface.co/spaces/presidio/presidio_demo)
    to detect Personally Identifiable Information (PII) in a given text.
    - With the example HF text dataset or the dataset of your own choice, you can get an idea of how much
    PII is contained in the dataset.
    - You can choose between spacy and HF NER models for PII detection and personalize which PII identity types
    should be included.
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
        "HuggingFaceFW/fineweb",
        "SetFit/bbc-news"
    ]
    st.selectbox("Examples", dataset_examples, key="_dataset", on_change=set_dataset)
    """
        You can also search for data sets on [HuggingFace Hub](https://huggingface.co/datasets).
        This repo supports data sets with data stored in json, jsonl, csv or parquet format.
    """

    # Enter a dataset and retrieve a list of data files
    dataset_name = st.text_input("Enter your dataset of interest", key='dataset')
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

    text_col_name = st.text_input('Name of the column with text is', 'text')

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

clean_df = clean_data(df, text_col_name)
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
# st_deny_allow_expander = st.expander(
#     "Allowlists and denylists",
#     expanded=False,
# )

# with st_deny_allow_expander:
#     st_allow_list = st_tags(
#         label="Add words to the allowlist", text="Enter word and press enter."
#     )
#     st.caption(
#         "Allowlists contain words that are not considered PII, but are detected as such."
#     )

#     st_deny_list = st_tags(
#         label="Add words to the denylist", text="Enter word and press enter."
#     )
#     st.caption(
#         "Denylists contain words that are considered PII, but are not detected as such."
#     )

st_entities_expander = st.expander("Choose entities to look for")
st_entities = st_entities_expander.multiselect(
    label="Which entities to look for?",
    options=get_supported_entities(*analyzer_params),
    default=list(get_supported_entities(*analyzer_params)),
    help="Limit the list of PII entities detected. "
    "This list is dynamic and based on the NER model and registered recognizers. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
)

# Calculate sample size.
population = clean_df.shape[0]
z_score = 1.96 # for 95% confidence level
st_dev = 0.5
margin_of_error = 0.05
numerator = (z_score**2 * st_dev * (1 - st_dev)) / margin_of_error**2
denominator = 1 + ((z_score**2 * st_dev * (1 - st_dev)) / (margin_of_error**2 * population))
sample_size = round(numerator/denominator)

# Prepare a sample
sample_df = clean_df.sample(n=sample_size, random_state=42)
sample_indices = sample_df.index
text_list = sample_df[text_col_name].values.tolist()
# df_dict = sample_df.to_dict(orient="list")

# Batch analyze
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

# Get the index of the sample the detected PII came from. 
len_list = [len(item) for item in st_batch_analyze_results]
index_list = [[i]*num for i, num in zip(sample_indices, len_list)]
index_list = [x for xs in index_list for x in xs]

# Add the corresponding index in the batch analyzer result dictionary.
# https://realpython.com/python-flatten-list/#flattening-a-list-using-standard-library-and-built-in-tools
flatten_results = [item.to_dict() for r in st_batch_analyze_results for item in r]
# https://www.geeksforgeeks.org/python-add-custom-values-key-in-list-of-dictionaries/
K = 'index_num'
flatten_results = list(map(lambda x, y:{**x, K:y}, flatten_results, index_list))

# Show the batch analyzer result as dataframe.
flatten_df = pd.DataFrame(flatten_results)
flatten_df = flatten_df.set_index('index_num')
results_df = pd.merge(flatten_df, sample_df, left_index=True, right_index=True)
results_df['identity'] = results_df.apply(
    lambda row: row['text'][row['start']:row['end']] if 0 <= row['start'] < len(row['text']) and 0 <= row['end'] <= len(row['text']) else '', 
    axis=1
)

# Sort the detected PII entity types and identities
count_df = results_df['entity_type'].value_counts()
name_df = results_df['identity'].value_counts()

n_pii = count_df.sum()
top_pii_type = count_df.index[0]
n_top_pii = count_df.iloc[0]
n_sample_with_pii = len(set(index_list))
per_sample_with_pii = round(n_sample_with_pii/sample_size*100,2)

st.markdown(
    f"The dataset contains :blue[{population}] records. "
    f"Using the sample size calculator, a total of :blue[{sample_size}] records were sampled. "
    f"The sampled dataset contains :blue[{n_pii}] PII entities. "
    f"The most frequently appearing PII entity type is :blue[{top_pii_type}], "
    f"which appeared :blue[{n_top_pii}] times. " 
    f":blue[{n_sample_with_pii}] sample records included PII, which means "
    f":blue[{per_sample_with_pii}]percentage of the sample dataset contains PII. "
)

with st.expander("Check which PII entity types and identities were detected"):

    fig, ax = plt.subplots(figsize=(3, 2))
    count_df.sort_values().plot.barh(ax=ax)
    ax.set_title("Most Frequently Detected PII Entity Types in the Dataset")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(3, 2))
    name_df.iloc[:10].sort_values().plot.barh(ax=ax)
    ax.set_title("Most Frequently Detected PII Entity Types in the Dataset")
    st.pyplot(fig)




