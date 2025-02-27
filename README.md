# Text Dataset Quality Evaluator

## Overview of the App

This Streamlit app helps responsible and reliable usage of text datasets by inspecting the quality of text datasets based on three criteria: gender bias, toxicity, and privacy breach. It is based on a Streamlit HuggingFace Connection app [1].

It supports HuggingFace text data sets with data stored in json, jsonl, csv or parquet format. 

It contains three demos for gender bias measurement, toxicity classification, and Personally Identifiable Information(PII) detection, which are described in detail below.

## Demos

Before each evaluation, a HuggingFace Text Dataset is selected and processed by dropping NA values, lowercasing, and dropping special characters. The dataset is then sampled with the appropriate sample size determined by [2]. All evaluation is then performed on the sampled dataset to provide a quick understanding on the aforementioned quality criterias of the text dataset. Gender imbalance is quantified as a score, which is explained below, and levels of toxicity and privacy breach are expressed with the number and percentage of records in the sampled dataset classified as toxic and with PII, respectively. These results could paint a better picture on if the dataset is appropriate and safe to use, in terms of the aforementioned criteria. 

### Gender Bias Measurement

Many studies have shown that gender bias reflected in the text data can propagate in the pre-trained word embeddings, LLMs, and NLP downstream tasks [3,4,5]. Therefore, it is important to at least recognize gender bias in the text dataset, for example by quantifying such bias, before it is used for any application. 

Though gender bias can have varying definitions, in the context of this app, it indicates imbalance of gender representations in the given text dataset. To quantify that, a gender magnitude metric suggested by [6] was used.

Specifically, gender magnitude of the dataset was defined using a set of pre-determined gender definitional words (e.g., she, woman, her, he, man, him, etc.) to quantify the degree of female-/male-related concepts. The pre-determined set of gender definitional words, as appeaered in the study, is provided in the app, but it can also be edited. The study used the term frequency(TF) of the words and the boolean value of whether the word exists in the dataset for calculating the gender magnitude. For example, the female magnitude of a dataset is defined as follows:

**TF**: 
$$mag^f(d) = \sum_{w \in G_f} \log \#(w, d)$$

**Boolean**: 
$$mag^f(d) =
\begin{cases}
1, & \text{if } \sum_{w \in G_f} \#(w, d) > 0 \\
0, & \text{otherwise}
\end{cases}$$

In the app, three options of calculating gender magnitude with total count, term frequency, and boolean values are offered. Final gender magnitude can be reported by male, female, or difference (female - male) metrics. 
Negative values in difference metric suggests gender imbalance in preference to the male concept in the dataset, and positive values vice versa.  

### Toxicity Classification

Motivated to encourage safe and effective online conversations without negative online behaviors (e.g. disrepectful and rude comments that might drive others away from the discussion), the Conversation AI team, a research initiative founded by Jigsaw and Google, held Kaggle challenges for toxic comment classification, unintended bias in toxic comments, and multilingual toxic comment classification in 2018, 2019, and 2020, respectively [7,8,9]. The goal was to train models to have a better performance than the pre-existing Perspective API toxicity classification models. In line of that effort, Unitary trained bert-base-uncased, roberta-base, and xlm-roberta-base models for those challenges, which were used in the app for toxicity classification [10]. 

In the app, each text was scored on the following labels for toxicity (toxic, severe_toxic, obscene, threat, insult, and identity_hate) and when the score for any label was above 0.5, it was classified as toxic. 


### PII Detection

Text dataset collected from documents, e-mails, chats, and social media platforms, etc, can contain private information, which might lead to a person's identification  or further potential harm [11]. Specifically, Personal Identifiable Information(PII) refers to any information that could show the identity of an individual, such as phone number, email address, IP address, and ZIP code. Exposing or sharing such sensitive information to a third party without the individual’s consent could lead to serious consequences, especially with the GDPR regulations, for the companies that operate in the European Union. Therefore, this app offers a way to check for PII in text using Microsoft Presidio [12], an open-source PII detection tool based on pattern detection and ML models. HuggingFace already offers this feature on the Dataset Hub, but this app helps check for PII information on any HuggingFace text datasets where it is not applied. 

This demo is built based on the Presidio Streamlit app [13] and offers options on Named Entity Recognition (NER) models from Spacy or HuggingFace. The specific PII categories that should be detected can be customized. 

## Demo App

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-examples.streamlit.app/) -->


## References

[1] [Streamlit HuggingFace Connection Demo](https://github.com/streamlit/files-connection/tree/main/hf-example)

[2] [Sample size calculator](https://www.surveymonkey.com/mp/sample-size-calculator/)

[3] Bolukbasi, T., Chang, K., Zou, J.Y., Saligrama, V., & Kalai, A.T. [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. Neural Information Processing Systems (2016)](https://arxiv.org/abs/1607.06520)

[4] De-Arteaga, M., Romanov, A., Wallach, H.M., Chayes, J.T., Borgs, C., Chouldechova, A., Geyik, S.C., Kenthapadi, K., & Kalai, A.T. [Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting. Proceedings of the Conference on Fairness, Accountability, and Transparency (2019)](https://arxiv.org/abs/1901.09451)

[5] Gallegos, I.O., Rossi, R.A., Barrow, J., Tanjim, M.M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., & Ahmed, N.  [Bias and Fairness in Large Language Models: A Survey. Computational Linguistics, 50, 1097-1179 (2023)](https://arxiv.org/abs/2309.00770)

[6] Rekabsaz, N., & Schedl, M. [Do Neural Ranking Models Intensify Gender Bias? Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (2020)](https://arxiv.org/abs/2005.00372)

[7] [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

[8] [Jigsaw Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

[9] [Jigsaw Multilingual Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)

[10] [Toxic Comment Classification Model on HuggingFace](https://huggingface.co/unitary/unbiased-toxic-roberta)

[11] Sousa, S., Kern, R. [How to keep text private? A systematic review of deep learning methods for privacy-preserving natural language processing. Artif Intell Rev 56, 1427–1492 (2023)](https://doi.org/10.1007/s10462-022-10204-6)

[12] [Microsoft Presidio App](https://huggingface.co/spaces/presidio/presidio_demo)

[13] [Experimenting with Automatic PII Detection on the Hub using Presidio](https://huggingface.co/blog/presidio-pii-detection)