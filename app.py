from dataclasses import replace
import streamlit as st
import functions
import numpy as np

# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1500}px
    }}
    .sidebar .sidebar-content {{
                width: 300px;
            }}
</style>

""",
        unsafe_allow_html=True,
    )

my_model = functions.read_model()
imdb_df = functions.read_data()

st.write("# Topic Modeling - IMDb Database")

st.sidebar.image('logo.png', width=200)
st.sidebar.write('## Menu options')
menu_options = st.sidebar.radio('', ['Explore topics', 'Search a movie'])


if menu_options == 'Explore topics':
    topic_count = my_model.get_topic_info()
    topic_count['Name'] = [i.replace('_', ' ') for i in topic_count['Name']]
    st.dataframe(topic_count)
    n_topics = st.slider('Number of topics in the plot', 4, 28, 12, 4)
    st.plotly_chart(my_model.visualize_barchart(top_n_topics=n_topics), use_container_width=True)

if menu_options == 'Search a movie':

    with st.form("my_form"):
        search_term = st.text_input('Type a search term')
        topics, probs = my_model.find_topics(search_term)
        topic_name = my_model.topic_names[topics[0]]
        topic_name = topic_name.replace('_', ' ')

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f'With {probs[0].round(3)} probability, the search term `"{search_term}"` falls into `"{topic_name}"` topic.')
            selected_movies = imdb_df[imdb_df.topic==topics[0]].reset_index(drop=True)
            how_many_movies = st.slider('Number of random movies from the topic', 4, 24, 8, 4)
            col1, col2, col3, col4 = st.columns(4)
            random_n = selected_movies.iloc[np.random.choice(selected_movies.shape[0], how_many_movies, replace=False),]

            with col1:
                for i in range(0, how_many_movies//4, 1):
                    st.write(f'###### [{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    st.text_area('', random_n.iloc[i].synopsis, height=150, key='1')
                    st.write(' ')

            with col2:
                for i in range(how_many_movies//4, 2*how_many_movies//4, 1):
                    st.write(f'###### [{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    st.text_area('', random_n.iloc[i].synopsis, height=150, key='2')
                    st.write(' ')

            with col3:
                for i in range(2*how_many_movies//4, 3*how_many_movies//4, 1):
                    st.write(f'###### [{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    st.text_area('', random_n.iloc[i].synopsis, height=150, key='3')
                    st.write(' ')

            with col4:
                for i in range(3*how_many_movies//4, 4*how_many_movies//4, 1):
                    st.write(f'###### [{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    st.text_area('', random_n.iloc[i].synopsis, height=150, key='4')
                    st.write(' ')

            with st.expander('Show all movies in the selected topic'):
                st.dataframe(selected_movies.drop('topic', axis=1))