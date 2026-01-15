import pandas as pd
import pickle
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load CSV
df = pd.read_csv("movies_metadata.csv")

# Basic cleaning
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title', 'overview', 'genres', 'tagline', 'vote_average', 'popularity']]
df = df.dropna(subset=['title'])

df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# Genres processing
df['genres'] = df['genres'].apply(
    lambda x: " ".join([i['name'] for i in ast.literal_eval(x)])
    if isinstance(x, str) else ""
)

# Combine text
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']

# NLP preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df['tags'] = df['tags'].apply(preprocess_text)
df = df.reset_index(drop=True)

# TF-IDF
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# Indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# SAVE PICKLES (LOCAL ENV)
df.to_pickle("df.pkl")
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl", "wb"))
pickle.dump(indices, open("indices.pkl", "wb"))

print("✅ Pickles created locally without compatibility issues")
