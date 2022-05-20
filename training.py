import pandas as pd
import numpy as np

import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords

import gensim
from gensim.utils import simple_preprocess

from bertopic import BERTopic
imdb_df = pd.read_csv('data/imdb_encoded.csv')
imdb_df.shape
imdb_df.head()
imdb_df.columns
## Data cleaning
# deleting rows with no synopsis
imdb_df = imdb_df[imdb_df['synopsis'] != 'Add a Plot'].dropna(subset=['synopsis']).reset_index(drop=True)
imdb_df.shape

# Removing punctuation
imdb_df['synopsis'] = imdb_df['synopsis'].map(lambda x: re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', x))
# Converting the text to lowercase
imdb_df['synopsis'] = imdb_df['synopsis'].map(lambda x: x.lower())
# Removing 'see full summary'
imdb_df['synopsis'] = imdb_df['synopsis'].map(lambda x: re.sub('see full summary\xa0»', '', x))
# Deleting unnecessary spaces
imdb_df['synopsis'] = imdb_df['synopsis'].str.strip()
# Lemmatization
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

imdb_df['synopsis_lemmatized'] = imdb_df['synopsis'].apply(lemmatize_text)

# Removing stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

data = imdb_df['synopsis_lemmatized'].values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
## BERTopic
model = BERTopic(min_topic_size=50, n_gram_range=(1,3), verbose=True)
docs = data_words
docs = np.array([(" ").join(i) for i in docs])
labels, probs = model.fit_transform(docs)
imdb_df['topic'] = labels
imdb_df.to_csv('data/ded_with_topics.csv', index=False)
model.visualize_barchart(top_n_topics=12)
model.save("models/topic_model", save_embedding_model=False)