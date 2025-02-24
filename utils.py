import streamlit as st
from pathlib import Path
import re
import pandas as pd
import numpy as np

@st.cache_data(ttl=3600)
def get_files(_conn, dataset):
    relevant_exts = ('**.csv', '**.jsonl', '**.parquet', '**.json')
    relevant_files = []
    for ext in relevant_exts:
        relevant_files.extend(_conn.fs.glob(Path(dataset, ext).as_posix()))
    return [f.replace(str(dataset) + '/', '') for f in relevant_files]

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("reviews_data.csv")
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

def clean_data(data):
    clean_data = data.dropna()
    return clean_data

def process_text(input_text):
    lowercase = input_text.lower() # lowercase
    replaced = re.sub(r'[^\w\s]', ' ', lowercase)  # remove any special characters
    return replaced

# def find_match_count(text: str, pattern: str) -> int:
#     return len(re.findall(pattern, text))

def find_match_count(text: str, pattern: str) -> int:
    return len(re.findall(r'\b' + pattern + r'\b', text))