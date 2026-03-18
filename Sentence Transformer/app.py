import streamlit as st

from deep_translator import GoogleTranslator
from langdetect import detect
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import faiss

import numpy as np
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.subheader("🔮 Prompt Bot (recomendación por texto/mood)")
st.caption("Primero aplicamos un filtro compuesto, luego usamos TF-IDF para recomendar dentro del subset filtrado.")

def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text
        return text
    except Exception as e:
        print(f"Error al detectar idioma: {e}")
        return text

# Clean developers and publishers
def clean_publisher_developer(dev):
    dev = dev.lower()      
    dev = dev.strip()       
    dev = re.sub(r'[.,]', '', dev)     
    dev = re.sub(r'\s+', ' ', dev)
    
    remove_words = ["inc", "ltd", "llc", "corp", "corporation", "company", "co"]
    
    words = dev.split()
    words = [w for w in words if w not in remove_words]
    
    return " ".join(words)

# Store model in cache
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


file_path = PROJECT_ROOT / "data" / "processed" / "df_clean.parquet"
faiss_index_path = PROJECT_ROOT / "data" / "processed" / "games_embeddings_faiss_IP.index"

df = pd.read_parquet(file_path)

df = df.reset_index(drop=True)
df["faiss_id"] = np.arange(len(df))

filtered_df = df.copy()

df["developers_clean"] = df["developers"].apply(lambda dev_list: [clean_publisher_developer(dev) for dev in dev_list])
df["publishers_clean"] = df["publishers"].apply(lambda publishers_list: [clean_publisher_developer(publisher) for publisher in publishers_list])

choosen_genres = []
choosen_categories = []
choosen_tags = []

choosen_developer = []
choosen_publisher = []

sorted_genres = sorted(set(g for sublist in df["genres"] for g in sublist))
sorted_categories = sorted(set(g for sublist in df["categories"] for g in sublist))
sorted_tags = sorted(set(g for sublist in df["tags"] for g in sublist))

choosen_genres = st.multiselect("Elige tus Género deseados", sorted_genres)
choosen_categories = st.multiselect("Elige tus Categorias deseadas", sorted_categories)
choosen_tags = st.multiselect("Elige tus Etiquetas deseadas ", sorted_tags)

show_dp = st.checkbox("Clicka el cuadrado para mostrar Desarrollador/Distribuidor", value=False)

if show_dp:
    sorted_developer = sorted(set(g for sublist in df["developers_clean"] for g in sublist))
    sorted_publisher = sorted(set(g for sublist in df["publishers_clean"] for g in sublist))

    choosen_developer = st.selectbox("Elige tu Desarrollador", sorted_developer)
    choosen_publisher = st.selectbox("Elige tu Distribuidor", sorted_publisher)


user_description = st.text_area("Para mejorar la recomendación escribe una descripción (máx 200 caracteres): ")
text=""

if len(user_description) > 200:
    st.warning(f"Has escrito {len(user_description)} caracteres. El máximo es 200.")
    text = user_description[:200]


bert_var = 0

if not choosen_genres and not choosen_categories and not choosen_tags and not choosen_developer and not choosen_publisher and not user_description:
    bert_var = 1

elif not user_description:
    bert_var = 2

    if choosen_genres:
        filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: any(g in x for g in choosen_genres))]

    if choosen_categories:
        filtered_df = filtered_df[filtered_df["categories"].apply(lambda x: any(g in x for g in choosen_categories))]

    if choosen_tags:
        filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: any(g in x for g in choosen_tags))]

    if choosen_developer:
        filtered_df = filtered_df[filtered_df["developers_clean"].apply(lambda x: choosen_developer in [g for g in x])]

    if choosen_publisher:
        filtered_df = filtered_df[filtered_df["publishers_clean"].apply(lambda x: choosen_publisher in [g for g in x])]

else:
    
    if choosen_genres:
        filtered_df = filtered_df[filtered_df["genres"].apply(lambda x: any(g in x for g in choosen_genres))]

    if choosen_categories:
        filtered_df = filtered_df[filtered_df["categories"].apply(lambda x: any(g in x for g in choosen_categories))]

    if choosen_tags:
        filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: any(g in x for g in choosen_tags))]

    if choosen_developer:
        filtered_df = filtered_df[filtered_df["developers_clean"].apply(lambda x: choosen_developer in [g for g in x])]

    if choosen_publisher:
        filtered_df = filtered_df[filtered_df["publishers_clean"].apply(lambda x: choosen_publisher in [g for g in x])]


k = st.slider('Cuantos juegos quieres de vuelta?', value = 5)

column_rename = {
    'name': 'Juego',
    'about_the_game': 'Descripción',
    'categories': 'Categorías',
    'genres': 'Géneros',
    'tags': 'Tags',
    'developers_clean': 'Desarrolladores',
    'publishers_clean': 'Distribuidores',
    'windows': 'Windows',
    'linux': 'Linux',
    'mac': 'Mac'
}

if bert_var==1:
    
    df["estimated_owners_max"] = df['estimated_owners'].str.split(' - ').str[1].astype(int)
    df_ordered_by_owners = df.sort_values(by='estimated_owners_max', ascending=False)
    df_display = df_ordered_by_owners[['name', 'about_the_game', 'categories', 'genres', 'tags', 'developers_clean', 'publishers_clean', 'windows', 'linux', 'mac']].head(k).rename(columns=column_rename).reset_index(drop=True)
    st.dataframe(df_display)

elif bert_var==2:

    filtered_df["estimated_owners_max"] = filtered_df['estimated_owners'].str.split(' - ').str[1].astype(int)
    filtered_df_ordered_by_owners = filtered_df.sort_values(by='estimated_owners_max', ascending=False)
    df_display = filtered_df_ordered_by_owners[['name', 'about_the_game', 'categories', 'genres', 'tags', 'developers_clean', 'publishers_clean', 'windows', 'linux', 'mac']].head(k).rename(columns=column_rename).reset_index(drop=True)
    st.dataframe(df_display)

else:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_translated = translate_to_english(user_description)
    model = load_model()

    query_embedding = model.encode([query_translated], device=device, normalize_embeddings=True)
    query_embedding_np = np.array(query_embedding, dtype=np.float32)

    index = faiss.read_index(str(faiss_index_path))

    distances, indices = index.search(query_embedding_np, 20000)

    filtered_indices = set(filtered_df["faiss_id"])

    if indices.size == 0:
        print("No se encontraron resultados.")
    else:
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in filtered_indices:
                sim_score = distances[0][i]
                results_df = df.iloc[idx].to_dict()
                results_df['similarity'] = sim_score
                results.append(results_df)

        results_df = pd.DataFrame(results)

    if results_df.empty:
        st.write("No hay resultados para esos filtros")
    else:
        st.write(results_df[['name', 'about_the_game', 'categories', 'genres', 'tags', 'developers_clean', 'publishers_clean', 'windows', 'linux', 'mac']].head(k).rename(columns=column_rename).reset_index(drop=True))
