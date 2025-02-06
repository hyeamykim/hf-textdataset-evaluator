# Is Your Text Dataset Gender Biased?

Streamlit App to evaluate the HuggingFace text datasets based on gender bias measurement, toxicity classification, and Personally Identifiable Information (PII) detection.

Inspired by 
- [Streamlit HF Dataset Explorer App](https://github.com/streamlit/files-connection/tree/main/hf-example)
- [Rekabsaz & Shedl(2020)](https://arxiv.org/pdf/2005.00372)
- Toxicity classification reference
- [Microsoft Presidio App](https://huggingface.co/spaces/presidio/presidio_demo)

## Overview of the App

This app was designed to encourage responsible and reliable usage of text datasets by inspecting the quality of text datasets based on three criteria: gender bias, toxicity, and PII. 

This repo supports HuggingFace text data sets with data stored in json, jsonl, csv or parquet format. 

### Gender Bias Measurement

First, the selected dataset goes through basic text processing of dropping NA values, lowercasing, and dropping special characters. Then, length and word count of each record in the dataset, and the average length and average word count of the dataset are calculated and presented. Thet 10 most common words in the dataset (except for stopwords) are presented. 

Then, based on the works of Rekabsaz & Shedl (2020), document gender magnitude was defined using a set of pre-determined gender definitional words (e.g., she, woman, her, he, man, him, etc.) to quantify the degree of female-/male-related concepts in a document. Two variants using the (sum of the logarithm of the) term frequency (TF) of the words and the boolean value of whether the word exists in the document can be calculated. For example, the female magnitude of a document is defined as follows:

**TF**: 
$$
mag^f(d) = \sum_{w \in G_f} \log \#(w, d)
$$

**Boolean**: 
$$ 
mag^f(d) =
\begin{cases}
1, & \text{if } \sum_{w \in G_f} \#(w, d) > 0 \\
0, & \text{otherwise}
\end{cases}
$$

The pre-determined set of gender definitional words is provided, but you can also add your own words.

### Toxicity Classification

### PII Detection



## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-examples.streamlit.app/)


## References

Rekabsaz, N., & Schedl, M. (2020). Do Neural Ranking Models Intensify Gender Bias? Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. (https://arxiv.org/pdf/2005.00372)

Streamlit HuggingFace Connection Demo (https://github.com/streamlit/files-connection/tree/main/hf-example)