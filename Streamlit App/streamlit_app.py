# -------------------------
# Setting Libraries
# -------------------------
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from deep_translator import GoogleTranslator
from langdetect import detect
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import faiss

import numpy as np

import sys
import os
import importlib.util

path = os.path.abspath(os.path.join(os.getcwd(), "Recommendation System", "user_api.py"))
spec = importlib.util.spec_from_file_location("user_api", path)
user_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_api)

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Videogame Recommender", page_icon="🎮", layout="wide")

DATA_URL = "https://huggingface.co/datasets/pabloramcos/Videogame-Recommender-Final-Project/resolve/main/games.parquet"

# Debug visual. Quítalo cuando ya no te haga falta.
st.write("VERSION CHECK:", "2026-03-18 20:30")


# -------------------------
# Helper functions
# -------------------------
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def pick_text_col(df: pd.DataFrame) -> str:
    """Elige la mejor columna de texto disponible."""
    if "detailed_description" in df.columns:
        return "detailed_description"
    if "short_description" in df.columns:
        return "short_description"
    return ""


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Devuelve la primera columna existente de una lista de posibles nombres."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def tokenize_field(value) -> list[str]:
    """
    Convierte una celda tipo:
    - 'Action|Indie'
    - "['Action', 'Indie']"
    - 'Action, Indie'
    en una lista limpia.
    """
    if value is None or pd.isna(value):
        return []

    s = str(value).strip()
    s = re.sub(r"[\[\]'\"]", "", s)
    parts = re.split(r"[|,;]", s)
    return [p.strip() for p in parts if p.strip()]


def build_options_from_col(df: pd.DataFrame, col: str, top_n: int = 150) -> list[str]:
    """Saca las opciones más frecuentes de una columna tipo genres/tags/categories."""
    if col not in df.columns:
        return []

    series = df[col].dropna().astype(str)
    tokens = series.apply(tokenize_field).explode()
    tokens = tokens.dropna()
    if len(tokens) == 0:
        return []

    vc = tokens.value_counts()
    return vc.head(top_n).index.tolist()


def apply_multi_select_filter(df_in: pd.DataFrame, col: str | None, selected: list[str]) -> pd.DataFrame:
    """
    Filtra el dataframe quedándose con filas donde la columna contiene
    cualquiera de los valores seleccionados.
    """
    if col is None or col not in df_in.columns or not selected:
        return df_in

    pattern = "|".join(re.escape(x) for x in selected)
    s = df_in[col].fillna("").astype(str)
    return df_in[s.str.contains(pattern, case=False, na=False)]


def apply_composite_filter(
    df_in: pd.DataFrame,
    genre_col: str | None,
    category_col: str | None,
    tag_col: str | None,
    platform_col: str | None,
    developer_col: str | None,
    publisher_col: str | None,
    selected_genres: list[str],
    selected_categories: list[str],
    selected_tags: list[str],
    selected_platforms: list[str],
    selected_developers: list[str],
    selected_publishers: list[str],
    include_kw: str,
    exclude_kw: str,
    text_col_for_kw: str
) -> pd.DataFrame:
    df2 = df_in.copy()

    # Filtros por columnas "taxonómicas"
    df2 = apply_multi_select_filter(df2, genre_col, selected_genres)
    df2 = apply_multi_select_filter(df2, category_col, selected_categories)
    df2 = apply_multi_select_filter(df2, tag_col, selected_tags)
    df2 = apply_multi_select_filter(df2, platform_col, selected_platforms)
    df2 = apply_multi_select_filter(df2, developer_col, selected_developers)
    df2 = apply_multi_select_filter(df2, publisher_col, selected_publishers)

    # Include / exclude keywords sobre el texto
    if text_col_for_kw and text_col_for_kw in df2.columns:
        txt = df2[text_col_for_kw].fillna("").astype(str)

        inc = [k.strip() for k in include_kw.split(",") if k.strip()]
        exc = [k.strip() for k in exclude_kw.split(",") if k.strip()]

        if inc:
            inc_pat = "|".join(re.escape(k) for k in inc)
            df2 = df2[txt.str.contains(inc_pat, case=False, na=False)]
            txt = df2[text_col_for_kw].fillna("").astype(str)

        if exc:
            exc_pat = "|".join(re.escape(k) for k in exc)
            df2 = df2[~txt.str.contains(exc_pat, case=False, na=False)]

    return df2

def fetch_user_profile_mock(steam_id: str) -> dict:
    """
    Placeholder temporal.
    Tus compañeros reemplazarán esto con la llamada real a la API.
    """
    return {
        "ok": True,
        "steam_id": steam_id,
        "user_name": "MockUser123",
        "is_public": True,
        "favorite_genres": ["Action", "RPG", "Indie"],
        "favorite_tags": ["Open World", "Story Rich", "Singleplayer"],
        "top_games": [
            {"name": "The Witcher 3", "hours": 120},
            {"name": "Hades", "hours": 85},
            {"name": "Skyrim", "hours": 200},
        ]
    }


def fetch_user_recommendations_mock(steam_id: str, top_k: int = 10) -> pd.DataFrame:

    data = [
        {"name": "Divinity: Original Sin 2", "genres": "RPG|Strategy", "reason": "Matches RPG and Story Rich preferences"},
        {"name": "Disco Elysium", "genres": "RPG|Narrative", "reason": "Strong story focus and singleplayer"},
        {"name": "Hollow Knight", "genres": "Action|Indie", "reason": "Matches Indie and exploration vibes"},
        {"name": "Outer Wilds", "genres": "Adventure|Exploration", "reason": "Fits story and discovery style"},
        {"name": "Baldur's Gate 3", "genres": "RPG|Strategy", "reason": "Strong overlap with favorite genres"},
    ]
    return pd.DataFrame(data).head(top_k)

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


# -------------------------
# Data loading (lazy + safe)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(url: str, sample_n: int = 5000) -> pd.DataFrame:
    try:
        df = pd.read_parquet(url)
    except Exception as e:
        raise RuntimeError(
            "No pude cargar el parquet desde HuggingFace.\n"
            "Posibles causas: sin internet, rate-limit, URL caída.\n"
            f"Detalle: {e}"
        )

    # Submuestreo para acelerar
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    # Normalizaciones típicas
    for c in ["name", "release_date", "short_description", "detailed_description"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    if "app_id" in df.columns:
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "required_age" in df.columns:
        df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

    return df


# -------------------------
# TF-IDF builders (cache)
# -------------------------
@st.cache_resource(show_spinner=False)
def build_tfidf_matrix(texts: tuple, max_features: int = 20000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(list(texts))
    return vec, X


# -------------------------
# App header
# -------------------------
st.title("🎮 Videogame Recommender")
st.caption("Explorador + recomendador TF-IDF + Prompt Bot + Chatbot.")

# Session state init
if "df" not in st.session_state:
    st.session_state["df"] = None

if "chat" not in st.session_state:
    st.session_state["chat"] = []

if "selected_row" not in st.session_state:
    st.session_state["selected_row"] = None


# -------------------------
# Sidebar: Load controls
# -------------------------
st.sidebar.header("Carga de datos")

sample_n_load = st.sidebar.slider("Tamaño de carga (rápido)", 1000, 30000, 5000, step=1000)

c1, c2 = st.sidebar.columns(2)
with c1:
    load_clicked = st.button("Cargar dataset")
with c2:
    reset_clicked = st.button("Reset")

if reset_clicked:
    st.session_state["df"] = None
    st.session_state["chat"] = []
    st.session_state["selected_row"] = None
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Reset hecho. Vuelve a cargar el dataset.")

if st.session_state["df"] is None:
    st.info("Pulsa **Cargar dataset** para empezar.")
    if load_clicked:
        try:
            with st.spinner("Cargando…"):
                st.session_state["df"] = load_data(DATA_URL, sample_n=sample_n_load)
            st.success("Dataset cargado ✅")
        except Exception as e:
            st.error(str(e))
    st.stop()

df = st.session_state["df"]


# -------------------------
# Dataset quick view
# -------------------------
with st.expander("📦 Vista rápida del dataset", expanded=False):
    st.write("Shape:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)


# -------------------------
# Global filters (apply to all modes)
# -------------------------
st.sidebar.header("Filtros globales")

q = st.sidebar.text_input("Buscar por nombre", "")

# Precio
price_range = None
if "price" in df.columns and df["price"].notna().any():
    pmin = float(df["price"].min())
    pmax = float(df["price"].max())
    if pd.notna(pmin) and pd.notna(pmax) and pmin <= pmax:
        price_range = st.sidebar.slider(
            "Precio",
            min_value=pmin,
            max_value=pmax,
            value=(pmin, min(pmax, 30.0)),
        )

# Edad
age_range = None
if "required_age" in df.columns and df["required_age"].notna().any():
    amin = int(df["required_age"].min())
    amax = int(df["required_age"].max())
    age_range = st.sidebar.slider("Edad requerida", min_value=amin, max_value=amax, value=(amin, amax))

# Año
year_range = st.sidebar.slider("Año (aprox)", min_value=1980, max_value=2026, value=(2005, 2026))

work = df.copy()

if q.strip() and "name" in work.columns:
    work = work[work["name"].str.contains(q, case=False, na=False)]

if price_range and "price" in work.columns:
    work = work[(work["price"] >= price_range[0]) & (work["price"] <= price_range[1])]

if age_range and "required_age" in work.columns:
    work = work[(work["required_age"] >= age_range[0]) & (work["required_age"] <= age_range[1])]

if "release_date" in work.columns:
    years = work["release_date"].astype(str).str.extract(r"(\d{4})")[0]
    work["_year"] = pd.to_numeric(years, errors="coerce")
    work = work[
        (work["_year"].fillna(year_range[0]) >= year_range[0]) &
        (work["_year"].fillna(year_range[1]) <= year_range[1])
    ]

st.write(f"Resultados (tras filtros globales): **{len(work):,}**")


# -------------------------
# Tabs / Modes
# -------------------------
st.markdown("## 🧠 Modos")
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Explorador", "🔮 Prompt Bot", "💬 Chatbot", "🆔 User recommender"])


# =========================================================
# TAB 1: Explorador + Recommender por juego
# =========================================================
with tab1:
    st.subheader("🔎 Explorador")

    cols_show = [c for c in ["app_id", "name", "release_date", "required_age", "price"] if c in work.columns]
    show = work[cols_show].head(1000)

    left, right = st.columns([1, 1])
    row = None

    with left:
        st.subheader("Lista")
        st.dataframe(show, use_container_width=True, height=540)

    with right:
        st.subheader("Ficha del juego")
        if len(show) == 0:
            st.info("No hay resultados con esos filtros.")
        else:
            selected_idx = st.selectbox(
                "Selecciona un juego",
                options=show.index.tolist(),
                format_func=lambda i: str(show.loc[i, "name"])[:80] if "name" in show.columns else str(i),
                key="selected_game_idx",
            )
            row = work.loc[selected_idx]
            st.session_state["selected_row"] = row.to_dict()

            st.markdown(f"### {row.get('name', '(sin nombre)')}")
            meta_cols = [c for c in ["app_id", "release_date", "required_age", "price"] if c in row.index]
            st.json({c: row.get(c) for c in meta_cols})

            desc = row.get("detailed_description", "") or row.get("short_description", "")
            if desc:
                st.markdown("**Descripción**")
                st.write(desc[:2000] + ("…" if len(desc) > 2000 else ""))

            if pd.notna(row.get("app_id", None)):
                st.markdown(f"**Steam:** https://store.steampowered.com/app/{int(row['app_id'])}/")

    st.divider()
    st.markdown("### 🤝 Recomendador simple (similares por descripción)")

    if row is None:
        st.info("Selecciona un juego arriba para ver recomendaciones.")
    else:
        cA, cB, cC = st.columns(3)
        with cA:
            sample_n_rec = st.slider("Tamaño muestra recomendador", 2000, 50000, 15000, step=1000, key="sample_n_rec")
        with cB:
            top_k = st.slider("Número de recomendaciones", 3, 20, 10, key="top_k_rec")
        with cC:
            max_feats = st.slider("Max features TF-IDF", 5000, 50000, 20000, step=5000, key="max_feats_rec")

        text_col = pick_text_col(work)
        if not text_col:
            st.info("No hay columna de texto para recomendar.")
        else:
            base = work.copy()
            if len(base) > sample_n_rec:
                base = base.sample(n=sample_n_rec, random_state=42)

            base[text_col] = base[text_col].fillna("").astype(str)

            if len(base) < 5 or base[text_col].str.len().sum() == 0:
                st.info("No hay texto suficiente para recomendar.")
            else:
                texts = tuple(base[text_col].tolist())
                _, X = build_tfidf_matrix(texts, max_features=max_feats)

                target_idx = None
                if "app_id" in base.columns and pd.notna(row.get("app_id", None)):
                    matches = base.index[base["app_id"] == row["app_id"]].tolist()
                    if matches:
                        target_idx = matches[0]

                if target_idx is None and "name" in base.columns:
                    matches = base.index[base["name"] == row["name"]].tolist()
                    if matches:
                        target_idx = matches[0]

                if target_idx is None:
                    st.info("El juego seleccionado no está en la muestra del recomendador.")
                else:
                    i = base.index.get_loc(target_idx)
                    sims = cosine_similarity(X[i], X).flatten()
                    order = sims.argsort()[::-1]
                    order = [j for j in order if j != i][:top_k]

                    recs_cols = [c for c in ["app_id", "name", "price", "release_date", "genres"] if c in base.columns]
                    recs = base.iloc[order][recs_cols].copy()
                    recs["similarity"] = sims[order]
                    st.dataframe(recs, use_container_width=True)


# =========================================================
# TAB 2: Prompt Bot
# =========================================================
with tab2:

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    st.subheader("🔮 Prompt Bot (recomendación por texto/mood)")
    st.caption("Primero aplicamos un filtro compuesto, luego usamos TF-IDF para recomendar dentro del subset filtrado.")

    # Store model in cache
    @st.cache_resource
    def load_model():
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    file_path = PROJECT_ROOT / "data" / "processed" / "df_clean.parquet"
    faiss_index_path = PROJECT_ROOT / "data" / "processed" / "games_embeddings_faiss_IP.index"

    df = pd.read_parquet(file_path)

    df = df.reset_index(drop=True)
    df["faiss_id"] = np.arange(len(df))
    
    
    df["developers_clean"] = df["developers"].apply(lambda dev_list: [clean_publisher_developer(dev) for dev in dev_list])
    df["publishers_clean"] = df["publishers"].apply(lambda publishers_list: [clean_publisher_developer(publisher) for publisher in publishers_list])

    filtered_df = df.copy()


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


    user_description = st.text_area("Para mejorar la recomendación escribe una descripción (máx 300 caracteres): ")
    text=""

    if len(user_description) > 300:
        st.warning(f"Has escrito {len(user_description)} caracteres. El máximo es 300 y solo leera hasta el último.")
        text = user_description[:300]



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
        
        if filtered_df.empty:
            st.write("No hay resultados para esos filtros")
        else:
            df["estimated_owners_max"] = df['estimated_owners'].str.split(' - ').str[1].astype(int)
            df_ordered_by_owners = df.sort_values(by='estimated_owners_max', ascending=False)
            df_display = df_ordered_by_owners[['name', 'about_the_game', 'categories', 'genres', 'tags', 'developers_clean', 'publishers_clean', 'windows', 'linux', 'mac']].head(k).rename(columns=column_rename).reset_index(drop=True)
            st.dataframe(df_display)

    elif bert_var==2:

        if filtered_df.empty:
            st.write("No hay resultados para esos filtros")
        else:
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



# =========================================================
# TAB 3: Chatbot
# =========================================================
with tab3:
    st.subheader("💬 Chatbot (Escribe 'ayuda' para ver ejemplos de preguntas)")
    st.caption("Chat básico para explicar el dataset, la app y el juego seleccionado.")

    def bot_reply(user_msg: str) -> str:
        msg = user_msg.lower().strip()

        if any(k in msg for k in ["help", "ayuda", "qué puedes", "que puedes", "commands", "comandos"]):
            return (
                "Puedo ayudarte con:\n"
                "- 'cuántos juegos hay' / 'shape'\n"
                "- 'columnas'\n"
                "- 'nulos'\n"
                "- 'top géneros' / 'top tags'\n"
                "- 'resumen del juego seleccionado'\n"
                "- 'cómo funciona tf-idf'\n"
            )

        if any(k in msg for k in ["cuántos", "cuantos", "shape", "tamaño", "tamano", "filas"]):
            return f"Ahora mismo hay {len(work):,} juegos tras filtros globales y {df.shape[1]} columnas en el dataset cargado."

        if any(k in msg for k in ["columnas", "columns", "campos"]):
            cols = list(df.columns)
            return "Columnas disponibles:\n- " + "\n- ".join(cols[:60]) + ("" if len(cols) <= 60 else "\n- ...")

        if any(k in msg for k in ["nulos", "missing", "nulls"]):
            nulls = (df.isna().mean() * 100).sort_values(ascending=False).head(10)
            lines = [f"{idx}: {val:.1f}%" for idx, val in nulls.items()]
            return "Top columnas con más nulos:\n" + "\n".join(lines)

        if any(k in msg for k in ["top", "géneros", "generos", "tags", "categorías", "categorias"]):
            col = first_existing_col(df, ["genres", "genre", "tags", "tag", "categories", "category"])
            if not col:
                return "No encuentro columna de géneros/tags/categorías en este dataset."

            series = df[col].dropna().astype(str)
            tokens = series.apply(tokenize_field).explode()
            tokens = tokens.dropna()
            top = tokens.value_counts().head(10)
            return "Top 10 valores:\n" + "\n".join([f"- {k} ({v})" for k, v in top.items()])

        if any(k in msg for k in ["seleccionado", "selected", "este juego", "resumen"]):
            row_dict = st.session_state.get("selected_row")
            if not row_dict:
                return "No hay juego seleccionado todavía. Ve a Explorador, elige uno y vuelve."

            name = row_dict.get("name", "(sin nombre)")
            price = row_dict.get("price", "N/A")
            desc = row_dict.get("short_description", "") or row_dict.get("detailed_description", "")
            desc = (desc[:400] + "…") if isinstance(desc, str) and len(desc) > 400 else desc
            return f"**{name}**\n\nPrecio: {price}\n\nDescripción: {desc}"

        if "tf-idf" in msg or "tfidf" in msg:
            return (
                "TF-IDF convierte textos en vectores según palabras importantes.\n"
                "Luego usamos similitud coseno para encontrar juegos con descripciones parecidas.\n"
                "Es un recomendador content-based, sin necesidad de datos de usuarios."
            )

        return "No pillé eso 😅. Escribe **'ayuda'** para ver ejemplos."

    for role, content in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Pregunta algo sobre el dataset o la app…")
    if user_input:
        st.session_state["chat"].append(("user", user_input))
        reply = bot_reply(user_input)
        st.session_state["chat"].append(("assistant", reply))
        st.rerun()

# =========================================================
# TAB 4: User Recommender
# =========================================================
with tab4:
    st.subheader("🆔 User Recommender")
    st.caption("Introduce un Steam ID para generar recomendaciones basadas en el perfil del usuario." \
    " Debe ser público.")

    url_user = st.text_input(
        "Steam ID del usuario, alias o url de perfil",
        placeholder="Ej: 7656119XXXXXXXXXX"
    )

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import ast

    COLUMNS = ["genres", "tags", "categories"]

    def clean_value(value):
        if value is None:
            return ""

        # NaN
        if isinstance(value, float) and np.isnan(value):
            return ""

        # ndarray
        if isinstance(value, np.ndarray):
            return " ".join(str(i).strip() for i in value if str(i).strip())

        # list
        if isinstance(value, list):
            return " ".join(str(i).strip() for i in value if str(i).strip())

        # dict
        if isinstance(value, dict):
            return " ".join(str(k).strip() for k in value.keys() if str(k).strip())

        # string
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text == "[]":
                return ""

            # 🔥 If it looks like a list or dict, try parsing
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed_value = ast.literal_eval(text)
                    if isinstance(parsed_value, dict):
                        return " ".join(str(k).strip() for k in parsed_value.keys() if str(k).strip())
                    if isinstance(parsed_value, list):
                        return " ".join(str(i).strip() for i in parsed_value if str(i).strip())
                except:
                    pass  # if parsing fails, continue

            return text

        return ""


    def remove_duplicates(text):
        words = text.split()
        return " ".join(dict.fromkeys(words))  # keeps order and removes duplicates

    def prepare_text(df):
        df = df.copy()
        clean_columns = []

        for col in COLUMNS:
            if col in df.columns:
                series = (
                    df[col]
                    .map(clean_value)   # faster than apply
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
                clean_columns.append(series)

        if not clean_columns:
            df["combined_features"] = ""
            return df

        # vectorized concatenation
        combined = clean_columns[0]
        for series in clean_columns[1:]:
            combined = combined.str.cat(series, sep=" ")

        # clean extra spaces
        df["combined_features"] = (
            combined
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .map(remove_duplicates)
        )

        return df


    def build_user_vector(df, user_games, X):

        if X is None:
            return None

        # Asegurarse de que app_id sea numérico
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")

        user_appids = user_games["appid"].astype(int).tolist()
        indices = df[df["app_id"].isin(user_appids)].index

        if len(indices) == 0:
            return None

        # Promedio de los vectores de los juegos del usuario
        user_vector = X[indices].mean(axis=0)

        # 🔥 Convertimos a array normal
        user_vector = np.asarray(user_vector)

        return user_vector

    def recommend(df, X, user_vector, user_games, top_n=10):

        if X is None or user_vector is None:
            print("❌ X o user_vector son None")
            return pd.DataFrame()

        # Asegurar que app_id sea numérico
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")

        # Calcular similitud
        similarities = cosine_similarity(user_vector, X)
        scores = similarities.flatten()

        df = df.copy()
        df["similarity"] = scores

        # 🔹 Obtener juegos del usuario
        user_appids = set(user_games["appid"].astype(int))

        # 🔹 No recomendar juegos que el usuario ya posee
        recommendations = df[~df["app_id"].isin(user_appids)]

        # Ordenar por similitud
        recommendations = recommendations.sort_values("similarity", ascending=False)

        return recommendations.head(top_n)



    import pandas as pd

    # 1️⃣ Cargar dataset
    print("🔄 Cargando dataset...")

    # 2️⃣ Preparar texto y matriz TF-IDF
    print("🧠 Preparando features...")
    df = prepare_text(df)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["combined_features"])

    # 3️⃣ Pedir URL de Steam
    steamid = user_api.extract_steamid(url_user)

    if not steamid:
        print("❌ No se pudo obtener el SteamID.")
        exit()

    print("✅ SteamID detectado:", steamid)

    # 4️⃣ Obtener juegos del usuario
    juegos_usuario = user_api.get_user_games(steamid)

    if juegos_usuario.empty:
        print("⚠️ Perfil privado o sin juegos.")
        exit()

    print(f"🎮 Juegos encontrados: {len(juegos_usuario)}")

    # 5️⃣ Construir vector del usuario
    user_vector = build_user_vector(df, juegos_usuario, X)

    if user_vector is None:
        print("⚠️ No se pudieron mapear los juegos al dataset.")
        exit()
    else: 
        print("Conseguido")

    # 6️⃣ Generar recomendaciones
    recomendaciones = recommend(df, X, user_vector, juegos_usuario, top_n=10)

    st.write("\n🎯 RECOMENDACIONES PERSONALIZADAS:\n")

    for i, row in recomendaciones.iterrows():
        st.write(f"🎮 {row['name']}")
        st.write(f"   Géneros: {row['genres']}")
        st.write(f"   Tags: {row['tags']}")
        st.write(f"   Categorias: {row['categories']}")


        st.write("-" * 30)

