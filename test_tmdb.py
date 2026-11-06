
import requests
import streamlit as st

api_key = st.secrets["TMDB_API_KEY"]

url = "https://api.themoviedb.org/3/movie/19995"

try:
    response = requests.get(
        url,
        params={"api_key": api_key},
        timeout=10
    )

    print("Status Code:", response.status_code)
    print("Response:", response.text[:500])

    if response.status_code == 200:
        poster_path = response.json().get("poster_path")

        if poster_path:
            print(
                "Poster URL:",
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )
        else:
            print("TMDB returned no poster.")

except requests.RequestException as e:
    print("Connection Error:", e)
