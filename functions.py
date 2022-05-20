import streamlit as st

import pandas as pd

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.backend._utils import select_backend

@st.cache(allow_output_mutation=True)
def read_model():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = select_backend(sentence_model)
    my_model = BERTopic.load('models/topic_model', embedding_model=model)
    return my_model

@st.cache
def read_data():
    imdb_df = pd.read_csv('data/ded_with_topics.csv')
    imdb_not_encoded = pd.read_csv('data/imdb_not_encoded.csv')
    imdb_clean = pd.merge(imdb_not_encoded[['synopsis', 'link']], imdb_df[['title', 'link', 'topic']], on='link')
    return imdb_clean

@st.cache(allow_output_mutation=True)
def vis_topics(model):
    return model.visualize_topics()

@st.cache(allow_output_mutation=True)
def vis_hierarchy(model):
    return model.visualize_hierarchy()





