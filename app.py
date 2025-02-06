import streamlit as st
from st_files_connection import FilesConnection
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from json import JSONDecodeError
from pathlib import Path
from utils import get_files

import re
from collections import Counter
import nltk
from nltk.corpus import stopwords


def main():

    st.set_page_config(
        page_title="Text Dataset Quality Check",
        page_icon=":white_check_mark:",
    )

    st.write('''
             # :white_check_mark: Text Dataset Quality Check
             '''
             )

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        The quality of the text datasets is increasingly important for any NLP downstream applications.
        This app helps you check for **Gender Bias, Toxicity, and Personally Identifiable Informationn (PII)**
        in any HuggingFace text dataset.

        **👈 Check out the demos from the sidebar** to see some examples!

        - **Gender Bias Demo**: It is well-known that gender bias in human-written texts get reflected
        and propagated in downstream NLP tasks (reference). One way to be better aware of gender bias
        in the text dataset is to measure the gender magnitude of the text data using a pre-defined
        set of gender defitional words and to check if male and female representations are balanced.
        The app also suggests a few ways of modification, if the given text dataset is gender-biased 
        (with unbalanced representation of male and female concepts), such as swapping the 
        over-represented gender words to their counterparts.

        - **Toxicity Demo**: Text that contains toxic language may have elements of hateful speech and 
        language others may find harmful or aggressive. Identifying toxic language is vital in tasks 
        such as content moderation and LLM training/evaluation, where appropriate action should be taken 
        to ensure safe platforms, chatbots, or other applications depending on this dataset.

        - **PII Demo**: 

        The datasets can be found at [HuggingFace](https://huggingface.co/datasets).
        
        You can check out the original [HF data set explorer app repo](https://github.com/streamlit/files-connection/tree/main/hf-example)..
        """
    )

    # st.subheader("Try with your own data set.", divider='grey')

    # uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # if uploaded_file is not None:

    #     file_name = uploaded_file.name

    #     try: 
    #         dataframe = pd.read_csv(uploaded_file,header='infer')
    #         st.write(dataframe)
    #     except:
    #         # sometimes encoding needs to be different
    #         with io.open(file_name, 'r', encoding='windows-1252') as data_file:
    #             dataframe = pd.read_csv(data_file, header=None)
    #             st.write(dataframe)
    
    # st.subheader(":hugging_face: HuggingFace Data Demo", divider='grey')
    # st.write(
    #     """
    #     This demo shows how many times gender definitional words such as he and she appear in 
    #     HF data sets in the gender statistics page.
        
    #     The datasets can be found at [HuggingFace](https://huggingface.co/datasets).
        
    #     You can check out the original [HF data set explorer app repo](https://github.com/streamlit/files-connection/tree/main/hf-example)..
    #     """
    # )

    # conn = st.experimental_connection('hf', type=FilesConnection)

    # with st.expander('Find dataset examples'):
    #     if 'dataset' not in st.session_state:
    #         st.session_state.dataset = "EleutherAI/lambada_openai"

    #     def set_dataset():
    #         st.session_state.dataset = st.session_state._dataset

    #     dataset_examples = [
    #         "EleutherAI/lambada_openai",
    #         "argilla/news-summary",
    #         "stanfordnlp/SHP",
    #         "HuggingFaceM4/tmp-pmd-synthetic-testing",
    #     ]
    #     st.selectbox("Examples", dataset_examples, key="_dataset", on_change=set_dataset)
    #     """
    #         You can also search for data sets on [HuggingFace Hub](https://huggingface.co/datasets).
    #         This repo supports data sets with data stored in json, jsonl, csv or parquet format.
    #     """

    # # Enter a dataset and retrieve a list of data files
    # dataset_name = st.text_input("Enter your dataset of interest", key='dataset')
    # dataset = Path('datasets', dataset_name)
    
    # dataset = Path('datasets',st.session_state.dataset)
    # dataset = dataset.as_posix()
    # file_names = get_files(conn, dataset)

    # if not file_names:
    #     st.warning(
    #         "No compatible data files found. This app only supports datasets stored in csv, json[l] or parquet format.")
    #     st.stop()

    # # Select a data file and row count to retrieve
    # file_selection = st.selectbox("Pick a data file", file_names)
    # datafile = Path(dataset, file_selection)
    # nrows = st.slider("Rows to retrieve", value=50)

    # "## Dataset Preview"
    # #kwargs = dict(nrows=nrows, ttl=3600)
    # kwargs = dict(ttl=3600)

    # # parquet doesn't neatly support nrows
    # # could be fixed with something like this:
    # # https://stackoverflow.com/a/69888274/20530083

    # #if datafile.suffix in ('.parquet', '.json'):
    # #    del (kwargs['nrows'])

    # try:
    #     df = conn.read(datafile.as_posix(), **kwargs)
    # except JSONDecodeError as e:
    #     # often times because a .json file is really .jsonl
    #     try:
    #         df = conn.read(datafile.as_posix(), input_format='jsonl', nrows=nrows, **kwargs)
    #     except:
    #         raise e

    # st.dataframe(df.head(nrows), use_container_width=True)

    # def clean_data(data):
    #     clean_data = data.dropna()
    #     return clean_data

    # def process_text(input_text):
    #     lowercase = input_text.lower() # lowercase
    #     replaced = re.sub(r'[^\w\s]', ' ', lowercase)  # remove any special characters
    #     return replaced

    # def find_match_count(text: str, pattern: str) -> int:
    #     return len(re.findall(r'\b' + pattern + r'\b', text))

    # clean_df = clean_data(df)
    # text_col_name = st.text_input('Name of the column with text is', 'text')
    # processed_df = clean_df[text_col_name].apply(lambda x: process_text(x))

    # gender_definitional_pairs = np.array([['he', 'she'],
    #                                         ['man', 'woman'],
    #                                         ['his', 'her'],
    #                                         ['himself', 'herself'],
    #                                         ['men', 'women'],
    #                                         ['husband', 'wife'],
    #                                         ['boy', 'girl'],
    #                                         ['brother', 'sister'],
    #                                         ['father', 'mother'],
    #                                         ['uncle', 'aunt'],
    #                                         ['grandfather', 'grandmother'],
    #                                         ['son', 'daughter'],
    #                                         ['nephew', 'niece'],
    #                                         ['boyfriend', 'girlfriend'],
    #                                         ['mr', 'ms'],
    #                                         ['papa', 'mama']], dtype='<U11')

    # male_words_list = list(gender_definitional_pairs[:, 0])
    # female_words_list = list(gender_definitional_pairs[:, 1])
    # gender_words_list = list(np.concatenate(gender_definitional_pairs))

        
    # '## EDA'

    # # length of text, word count, avg length, avg word count 
    # copy_df = pd.DataFrame(processed_df, columns=[text_col_name])

    # copy_df['length'] = copy_df[text_col_name].str.len()   
    # copy_df['wordcount'] = copy_df[text_col_name].str.split().apply(len)
    # avg_length = round(np.mean(copy_df['length']))
    # avg_word_count = round(np.mean(copy_df['wordcount']))

    # def visualize(col):
        
    #     fig, ax = plt.subplots(tight_layout=True)
    #     ax.hist(copy_df[col])
    #     ax.set_title(f"Histogram of {col}")
    #     st.pyplot(fig)

    # # 10 most frequent words
    # nltk.download('stopwords')
    # stop=set(stopwords.words('english'))

    # corpus=[]
    # copy_df['wordlist'] = copy_df[text_col_name].str.split()
    # word_list = copy_df['wordlist'].values.tolist()
    # corpus=[word for i in word_list for word in i if word not in stop]
    # total_word_count = len(corpus)

    # mostCommon = Counter(corpus).most_common(10)

    # words = []
    # freq = []
    # for word, count in mostCommon:
    #     words.append(word)
    #     freq.append(count)

    # ## generate this report/summary with other language models
    # st.markdown(
    #     "From the dataset, null values were dropped, text was turned into lowercase, "
    #     "and special charcters were removed. "
    #     f"The average text length is :blue[{avg_length}] " 
    #     f"and the average word count is :blue[{avg_word_count}]. "
    #     "The total number of words in the corpus except for stopwords is "
    #     f":blue[{total_word_count}]. "
    #     f"The 10 most frequently used words in the text are :blue[{words}]."
    # )

    # # length, word count distribution plot
    # features = copy_df.columns.tolist()[1:3]
    # for feature in features:
    #     visualize(feature)

    # # 10 most frequent word plot
    # fig, ax = plt.subplots()    
    # sns.barplot(x=freq, y=words)
    # plt.title('Top 10 Most Frequently Occuring Words')
    # st.pyplot(fig)
    
    # st.subheader("Gender Magnitude of the Dataset", divider='grey')
    # options = st.multiselect(
    #     'Pre-defined gender-definitional words:',
    #     gender_words_list,
    #     gender_words_list)
    # new_element = st.text_input('Add new gender-definitional words:', '')
    # options.append(new_element)

    # copy_df = pd.DataFrame(processed_df, columns=[text_col_name])

    # for col in options:
    #     if col != '':
    #         copy_df[col] = copy_df[text_col_name].apply(find_match_count, pattern=col)

    # updated_male_list = [w for w in male_words_list if w in options]
    # updated_female_list = [w for w in female_words_list if w in options]

    # copy_df['male_count'] = copy_df[updated_male_list].sum(axis=1)
    # copy_df['female_count'] = copy_df[updated_female_list].sum(axis=1)

    # gender_counts = pd.DataFrame(copy_df.iloc[:, 1:-2].sum().sort_values(ascending=False), columns=['count'])
    # # st.write(gender_counts)

    # n_gender_words = gender_counts['count'].sum()
    # top_gender_word = gender_counts.index[0]
    # n_top_gender_word = gender_counts['count'].iloc[0]
    # n_male_words = copy_df['male_count'].sum()
    # n_female_words = copy_df['female_count'].sum()

    # def get_tokens(text):
    #     return text.lower().split(" ")

    # def get_bias(tokens):
    #     text_cnt = Counter(tokens)

    #     cnt_feml = 0
    #     cnt_male = 0
    #     cnt_logfeml = 0
    #     cnt_logmale = 0
    #     for word in text_cnt:
    #         if word in updated_female_list:
    #             cnt_feml += text_cnt[word]
    #             cnt_logfeml += np.log(text_cnt[word] + 1)
    #         elif word in updated_male_list:
    #             cnt_male += text_cnt[word]
    #             cnt_logmale += np.log(text_cnt[word] + 1)
    #     text_len = np.sum(list(text_cnt.values()))

    #     bias_tc = (float(cnt_feml - cnt_male), float(cnt_feml), float(cnt_male))
    #     bias_tf = (np.log(cnt_feml + 1) - np.log(cnt_male + 1), np.log(cnt_feml + 1), np.log(cnt_male + 1))
    #     bias_bool = (np.sign(cnt_feml) - np.sign(cnt_male), np.sign(cnt_feml), np.sign(cnt_male))

    #     return bias_tc, bias_tf, bias_bool

    #st.markdown(
    #    get_bias (get_tokens("a war day and many boys , women and men"))
    #)

    # st.markdown(
    #     f"The dataset contains a total of :blue[{n_gender_words}] gender-definitional words. "
    #     f"The most frequently appearing gender word is :blue[{top_gender_word}], "
    #     f"which appeared :blue[{n_top_gender_word}] times. " 
    #     f"The female associated words appeared :blue[{n_female_words}] times and "
    #     f"The male associated words appeared :blue[{n_male_words}] times in the dataset. "
    # )

    # fig, ax = plt.subplots(figsize=(6, 4))
    # gender_counts.sort_values(by='count').plot.barh(ax=ax)
    # ax.set_title("Most Frequently used gender definitional words in the data set")
    # st.pyplot(fig)

    # bins = 10
    # fig, ax = plt.subplots(figsize=(5,4))
    # ax.hist(copy_df['female_count'], bins, alpha=0.5, label='female')
    # ax.hist(copy_df['male_count'], bins, alpha=0.5, label='male')
    # ax.legend(loc='upper right')
    # ax.set_title('Histogram of male- and female-definitional words in the data set')
    # st.pyplot(fig)

    # specific_options = st.multiselect(
    #     'If you want to check the distribution of specific gender words, choose from below.',
    #     gender_words_list,
    #     ['he','she'])

    # if len(specific_options)!=0:
    #     fig, axs = plt.subplots()
    #     for idx, w in enumerate(specific_options):
    #         axs.hist(copy_df[w], bins, alpha=0.5, label=w)
    #     axs.legend(loc='upper right')
    #     axs.set_title(f'Histogram of {specific_options} in the data set')
    #     st.pyplot(fig)
    

if __name__ == '__main__':
    main()

