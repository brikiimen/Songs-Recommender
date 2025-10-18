import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel
import re

# Load pre-trained model and data 
@st.cache_resource
def load_model_and_data():
    tfidf = joblib.load("tfidf_model.joblib")
    tfidf_matrix = joblib.load("tfidf_matrix.joblib")
    songs = pd.read_csv("songs.csv")
    return tfidf, tfidf_matrix, songs

tfidf, tfidf_matrix, songs = load_model_and_data()

# Streamlit UI 
st.title("🎧 Music Recommendation System")

st.header("Tell me what you like 🎶")
col1, col2 = st.columns(2)
with col1:
    preferred_genres = st.text_input("Preferred genres (comma-separated)", value="indie pop, alternative")
    favorite_artists = st.text_input("Favorite artists", value="Billie Eilish")
with col2:
    liked_songs = st.text_input("Songs you like (semicolon-separated)", value="Ocean Eyes; The Less I Know The Better")
    mood = st.text_input("Mood / vibe (e.g., calm, energetic)", value="dreamy")

n_recs = st.slider("Number of recommendations", 1, 20, 8)

#  Build user text
def build_user_text(genres_str, artists_str, liked_str, mood_str, keywords_str=""):
    parts = []
    if genres_str:
        parts += [g.strip() for g in genres_str.split(',') if g.strip()]
    if artists_str:
        parts += [a.strip() for a in artists_str.split(',') if a.strip()]
    if liked_str:
        parts += [s.strip() for s in re.split(r'[;,]', liked_str) if s.strip()]
    if mood_str:
        parts.append(mood_str)
    if keywords_str:
        parts.append(keywords_str)
    return " ".join(parts)

# Recommend 
if st.button("Recommend songs"):
    user_text = build_user_text(preferred_genres, favorite_artists, liked_songs, mood)
    user_vec = tfidf.transform([user_text])
    cosine_sim_user = linear_kernel(user_vec, tfidf_matrix).flatten()

    # genre boost
    for g in [g.strip() for g in preferred_genres.split(',') if g.strip()]:
        mask = songs['Genre'].str.contains(g, case=False, na=False)
        cosine_sim_user += mask.astype(float) * 0.12

    # Exclude liked songs
    liked_titles = set([t.strip().lower() for t in re.split(r'[;,]', liked_songs) if t.strip()])
    indices = cosine_sim_user.argsort()[::-1]

    recommendations = []
    for idx in indices:
        title = songs.iloc[idx]['Song']
        if title.lower() in liked_titles:
            continue
        recommendations.append((idx, cosine_sim_user[idx]))
        if len(recommendations) >= n_recs:
            break

    rec_indices = [r[0] for r in recommendations]
    df_recs = songs.iloc[rec_indices].copy()
    df_recs['score'] = [r[1] for r in recommendations]

    st.subheader("🎵 Recommended songs:")
    st.table(df_recs[['Song', 'Artist', 'Genre', 'score']])
