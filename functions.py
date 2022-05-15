import streamlit as st

import pandas as pd

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.backend._utils import select_backend

@st.cache
def read_model():
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    model = select_backend(sentence_model)
    my_model = BERTopic.load('models/topic_model', embedding_model=model)
    return my_model

@st.cache
def read_data():
    imdb_df = pd.read_csv('data/imdb_encoded_with_topics.csv')
    return imdb_df[['synopsis', 'topic', 'link']]

