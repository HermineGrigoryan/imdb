import streamlit as st
import functions

model = functions.read_model()
imdb_df = functions.read_data()

st.write(imdb_df.columns)