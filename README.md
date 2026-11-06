# CineMatch — Movie Recommendation System

A Netflix-inspired movie discovery website built with Streamlit and a TF-IDF/cosine-similarity recommendation model trained in Kaggle.

## Export your Kaggle model

Run **after** your `df` and `tfidf_matrix` have been created and the `release_year` and `genre_names` columns have been added. The row order must match the TF-IDF matrix. If you merged metadata after creating the matrix, verify that rows have not been reordered or duplicated; ideally regenerate the matrix from the final `df` before exporting.

```python
from scipy.sparse import save_npz
import os

assert len(df) == tfidf_matrix.shape[0]
assert {'id', 'title', 'release_year', 'genre_names', 'overview'}.issubset(df.columns)

os.makedirs('/kaggle/working/data', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)
df.reset_index(drop=True).to_pickle('/kaggle/working/data/movies.pkl')
save_npz('/kaggle/working/models/tfidf_matrix.npz', tfidf_matrix)
print('Export complete')
```

Download both files from Kaggle's `/kaggle/working/` output and put them in this project's `data/` and `models/` folders.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Movie posters

Create `.streamlit/secrets.toml` locally with:

```toml
TMDB_API_KEY = "your_actual_tmdb_v3_key"
```

Do **not** commit this file to GitHub. You can also set `TMDB_API_KEY` as an environment variable or in Streamlit Community Cloud secrets. Without the key or Internet, the app still recommends movies and displays poster placeholders.

## Notes

- The search input filters titles and the select box lets you choose one of the first 30 matches.
- Similarity scores are TF-IDF cosine similarity values × 100, not probabilities of liking a movie.
- This product uses the TMDB API but is not endorsed or certified by TMDB. Follow TMDB attribution requirements when publishing.
