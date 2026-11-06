import os
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CineMatch | Movie Discovery', page_icon='🎬', layout='wide')
st.markdown('''<style>
.stApp {background: #101014; color: #f7f7f7;}
.block-container {padding-top: 1.5rem;}
h1,h2,h3 {color: #f8f8f8;}
div.stButton > button {background:#e50914;color:white;border:0;border-radius:8px;font-weight:700;}
.movie-meta {color:#b8b8c0;font-size:0.88rem;}
.score {color:#55df98;font-weight:700;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    df = pd.read_pickle(ROOT / 'data' / 'movies.pkl')
    matrix = load_npz(ROOT / 'models' / 'tfidf_matrix.npz')
    if len(df) != matrix.shape[0]:
        raise ValueError('Movie metadata and TF-IDF matrix rows do not match. Re-export from Kaggle.')
    return df.reset_index(drop=True), matrix

@st.cache_data(ttl=86400, show_spinner=False)
def poster_for(movie_id):
    try:
        api_key = st.secrets.get('TMDB_API_KEY', os.getenv('TMDB_API_KEY', ''))
    except Exception:
        api_key = os.getenv('TMDB_API_KEY', '')
    if not api_key:
        return None
    try:
        response = requests.get(f'https://api.themoviedb.org/3/movie/{int(movie_id)}',
                                params={'api_key': api_key}, timeout=7)
        response.raise_for_status()
        path = response.json().get('poster_path')
        return f'https://image.tmdb.org/t/p/w500{path}' if path else None
    except (requests.RequestException, ValueError):
        return None

def recommendations(df, matrix, movie_idx, n=5):
    scores = linear_kernel(matrix[movie_idx], matrix).ravel()
    order = scores.argsort()[::-1]
    order = [i for i in order if i != movie_idx and df.iloc[i]['id'] != df.iloc[movie_idx]['id']][:n]
    return [(df.iloc[i], float(scores[i])) for i in order]

st.title('🎬 CineMatch')
st.caption('Discover your next favorite movie · Content-based AI recommendations')
try:
    df, matrix = load_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(f'Model files missing or invalid: {exc}. Export the files from Kaggle first.')
    st.stop()

names = df['title'].fillna('').astype(str)
query = st.text_input('Search for a movie', placeholder='Start typing: Iron Man, Avatar, Interstellar…')
if query.strip():
    matched = df[names.str.contains(query.strip(), case=False, regex=False, na=False)]
else:
    matched = df.head(50)
if matched.empty:
    st.warning('No matching titles found. Try another search.')
    st.stop()
choices = matched.head(30)
selection = st.selectbox('Choose a movie', choices.index.tolist(),
                         format_func=lambda idx: f"{df.loc[idx, 'title']} ({int(df.loc[idx, 'release_year']) if pd.notna(df.loc[idx, 'release_year']) else 'Unknown'})")
if st.button('✨ Get Recommendations', use_container_width=True):
    st.session_state['chosen_movie'] = int(selection)
if 'chosen_movie' in st.session_state:
    chosen = st.session_state['chosen_movie']
    st.subheader(f"Because you liked {df.iloc[chosen]['title']}")
    cols = st.columns(5)
    for col, (movie, score) in zip(cols, recommendations(df, matrix, chosen)):
        with col:
            poster = poster_for(movie['id'])
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown('🎞️ *Poster unavailable*')
            st.markdown(f"**{movie['title']}**")
            year = str(int(movie['release_year'])) if pd.notna(movie.get('release_year')) else 'Unknown'
            genres = movie.get('genre_names', 'Unknown')
            st.caption(f"{year} · {genres if pd.notna(genres) else 'Unknown'}")
            st.markdown(f"<span class='score'>{score*100:.1f}% similarity</span>", unsafe_allow_html=True)
            overview = movie.get('overview', '')
            if isinstance(overview, str) and overview.strip():
                with st.expander('Synopsis'):
                    st.write(overview)
st.divider()
st.caption('Movie metadata and posters: TMDB. Similarity is a text-feature score, not a predicted user rating. This product is not endorsed or certified by TMDB.')
