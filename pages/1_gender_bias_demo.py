
import streamlit as st
# from st_files_connection import FilesConnection
from json import JSONDecodeError
from pathlib import Path
from utils import get_files, clean_data, process_text, find_match_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Gender Bias Demo", page_icon=":female_sign:")
st.markdown("# :female_sign: Gender Bias Demo")
st.info(
    """
    - This demo is built on top of the [HF data set explorer app](https://github.com/streamlit/files-connection/tree/main/hf-example).
    - The gender magnitude metric devised by [Rekabsaz & Schedl(2020)](https://arxiv.org/abs/2005.00372)
    was used to test if there is gender imabalance in a given text. 
    - The gender definitional words used to calculate the metric can be personalized. 
    - It also shows how many times gender definitional words such as he and she appeared
     and how they are distributed in the dataset.
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

gender_definitional_pairs = np.array([['he', 'she'],
                                        ['man', 'woman'],
                                        ['his', 'her'],
                                        ['himself', 'herself'],
                                        ['men', 'women'],
                                        ['husband', 'wife'],
                                        ['boy', 'girl'],
                                        ['brother', 'sister'],
                                        ['father', 'mother'],
                                        ['uncle', 'aunt'],
                                        ['grandfather', 'grandmother'],
                                        ['son', 'daughter'],
                                        ['nephew', 'niece'],
                                        ['boyfriend', 'girlfriend'],
                                        ['mr', 'ms'],
                                        ['papa', 'mama']], dtype='<U11')

male_words_list = list(gender_definitional_pairs[:, 0])
female_words_list = list(gender_definitional_pairs[:, 1])
gender_words_list = list(np.concatenate(gender_definitional_pairs))

st.subheader("Gender Magnitude of the Dataset", divider='grey')

st.markdown(
    '''
    Following the works of Rekabsaz & Shedl (2020), gender magnitude in the text dataset can be measured 
    based on the presence of pre-defined gender-definitional words (e.g., she, woman, her, he, man, him). 
    
    The calculation involves some sort of combining or summarizing gender concepts represented in the 
    text by counting the gender-definitional words.
    For couting, three variants--total count, term frequency, and boolean value--can be used. 

    The female magnitude of a text can be defined using feminine concept words and the male magnitude
    of a text using male concept words. The difference of those two magnitudes (e.g. female magnitude - 
    male magnitude) can also be used as a metric.

    The gender magnitude of the entire dataset can be obtained by taking the average of the gender
    magnitude values of each text data.
    '''
)

# Total count shows the total number of times each gender-definitional word appeared in the text.
# Term frequency indicates
# Boolean value shows whether gender-definitional words existed in the text or not.

gender_words = st.multiselect(
    'Pre-defined gender-definitional words:',
    gender_words_list,
    gender_words_list)
new_element = st.text_input('Add new gender-definitional words:', '')
gender_words.append(new_element)

# copy_df = pd.DataFrame(processed_df, columns=[text_col_name])
copy_df = pd.DataFrame(sample_df, columns=[text_col_name])

for col in gender_words:
    if col != '':
        copy_df[col] = copy_df[text_col_name].apply(find_match_count, pattern=col)

updated_male_list = [w for w in male_words_list if w in gender_words]
updated_female_list = [w for w in female_words_list if w in gender_words]

copy_df['male_count'] = copy_df[updated_male_list].sum(axis=1)
copy_df['female_count'] = copy_df[updated_female_list].sum(axis=1)

gender_counts = pd.DataFrame(copy_df.iloc[:, 1:-2].sum().sort_values(ascending=False), columns=['count'])

n_gender_words = gender_counts['count'].sum()
top_gender_word = gender_counts.index[0]
n_top_gender_word = gender_counts['count'].iloc[0]
n_male_words = copy_df['male_count'].sum()
n_female_words = copy_df['female_count'].sum()

def get_tokens(text):
    return text.split(" ")

def get_bias(tokens):
    text_cnt = Counter(tokens)

    cnt_feml = 0
    cnt_male = 0
    cnt_logfeml = 0
    cnt_logmale = 0
    for word in text_cnt:
        if word in updated_female_list:
            cnt_feml += text_cnt[word]
            cnt_logfeml += np.log(text_cnt[word] + 1)
        elif word in updated_male_list:
            cnt_male += text_cnt[word]
            cnt_logmale += np.log(text_cnt[word] + 1)
    text_len = np.sum(list(text_cnt.values()))

    bias_tc = (float(cnt_feml - cnt_male), float(cnt_feml), float(cnt_male))
    bias_tf = (np.log(cnt_feml + 1) - np.log(cnt_male + 1), np.log(cnt_feml + 1), np.log(cnt_male + 1))
    bias_bool = (np.sign(cnt_feml) - np.sign(cnt_male), np.sign(cnt_feml), np.sign(cnt_male))

    return bias_tc, bias_tf, bias_bool


tokens_df = copy_df[text_col_name].to_frame()
tokens_df['tokens'] = tokens_df[text_col_name].apply(get_tokens)
tokens_df.drop(columns = text_col_name, inplace=True)
tokens_df[['bias_tc', 'bias_tf', 'bias_bool']] = pd.DataFrame(tokens_df['tokens'].apply(get_bias).tolist(), index=tokens_df.index)


calc_option = st.selectbox(
    "Which metric would you like to use to calculate gender magnitude?",
    ("Total Count", "Term Frequency", "Boolean"),
)
calc_option_dict = {"Total Count": 'bias_tc', "Term Frequency": 'bias_tf', "Boolean": 'bias_bool'}

met_option = st.selectbox(
    "Which metric would you like to use as gender magnitude?",
    ("F-M", "F", "M")
)
met_option_dict = {"F-M": 0, "F": 1, "M": 2}

bias_col = tokens_df[calc_option_dict[calc_option]]
diff_bias_values = [t[met_option_dict[met_option]] for t in bias_col]
mean_bias = round(np.mean(diff_bias_values),2)

st.markdown(
    f"The dataset contains :blue[{population}] records. "
    f"Using the sample size calculator, a total of :blue[{sample_size}] records were sampled. "
    f"The sampled dataset contains a total of :blue[{n_gender_words}] gender-definitional words. "
    f"The most frequently appearing gender word is :blue[{top_gender_word}], "
    f"which appeared :blue[{n_top_gender_word}] times. " 
    f"The female associated words appeared :blue[{n_female_words}] times and "
    f"The male associated words appeared :blue[{n_male_words}] times in the dataset. "
    f"The gender magnitude({met_option}) of the dataset is :blue[{mean_bias}]."
)

with st.expander("Check which gender-definitional words were detected"):

    fig, ax = plt.subplots(figsize=(6, 4))
    gender_counts.sort_values(by='count').plot.barh(ax=ax)
    ax.set_title("Most Frequently used Gender-definitional Words in the Dataset")
    st.pyplot(fig)

    bins = 10
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(copy_df['female_count'], bins, alpha=0.5, label='female')
    ax.hist(copy_df['male_count'], bins, alpha=0.5, label='male')
    ax.legend(loc='upper right')
    ax.set_title('Histogram of Male- and Female-definitional Words in the Dataset')
    st.pyplot(fig)

    specific_options = st.multiselect(
        'If you want to check the distribution of specific gender words, choose from below.',
        gender_words_list,
        ['he','she'])

    if len(specific_options)!=0:
        fig, axs = plt.subplots()
        for idx, w in enumerate(specific_options):
            axs.hist(copy_df[w], bins, alpha=0.5, label=w)
        axs.legend(loc='upper right')
        axs.set_title(f'Histogram of {specific_options} in the data set')
        st.pyplot(fig)