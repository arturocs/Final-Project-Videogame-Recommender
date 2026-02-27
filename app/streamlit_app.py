#Setting Libraries
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Videogame Recommender", page_icon="🎮", layout="wide")

DATA_URL = "https://huggingface.co/datasets/pabloramcos/Videogame-Recommender-Final-Project/resolve/main/games.parquet"


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
# App header
# -------------------------
st.title("🎮 Videogame Recommender")
st.caption("Explorador + recomendador simple (TF-IDF) usando el dataset en parquet de HuggingFace.")

# Session state init
if "df" not in st.session_state:
    st.session_state["df"] = None

# Sidebar: load controls
st.sidebar.header("Carga de datos")

sample_n = st.sidebar.slider("Tamaño de carga (rápido)", 1000, 30000, 5000, step=1000)

c1, c2 = st.sidebar.columns(2)
with c1:
    load_clicked = st.button("Cargar dataset")
with c2:
    reset_clicked = st.button("Reset")

if reset_clicked:
    st.session_state["df"] = None
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Reset hecho. Vuelve a cargar el dataset.")

if st.session_state["df"] is None:
    st.info("Pulsa **Cargar dataset** para empezar (esto evita el 502 al arrancar).")
    if load_clicked:
        try:
            with st.spinner("Cargando…"):
                st.session_state["df"] = load_data(DATA_URL, sample_n=sample_n)
            st.success("Dataset cargado ✅")
        except Exception as e:
            st.error(str(e))
    st.stop()

df = st.session_state["df"]

with st.expander("📦 Vista rápida del dataset", expanded=False):
    st.write("Shape:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------
# Filters
# -------------------------
st.sidebar.header("Filtros")
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

# Año (best-effort usando release_date)
year_range = st.sidebar.slider("Año (aprox)", min_value=1980, max_value=2026, value=(2005, 2026))

work = df.copy()

if q.strip() and "name" in work.columns:
    work = work[work["name"].str.contains(q, case=False, na=False)]

if price_range and "price" in work.columns:
    work = work[(work["price"] >= price_range[0]) & (work["price"] <= price_range[1])]

if age_range and "required_age" in work.columns:
    work = work[(work["required_age"] >= age_range[0]) & (work["required_age"] <= age_range[1])]

if "release_date" in work.columns:
    years = work["release_date"].str.extract(r"(\d{4})")[0]
    work["_year"] = pd.to_numeric(years, errors="coerce")
    work = work[
        (work["_year"].fillna(year_range[0]) >= year_range[0]) &
        (work["_year"].fillna(year_range[1]) <= year_range[1])
    ]

st.write(f"Resultados: **{len(work):,}**")

cols_show = [c for c in ["app_id", "name", "release_date", "required_age", "price"] if c in work.columns]
show = work[cols_show].head(1000)

left, right = st.columns([1, 1])

row = None  # por defecto no hay juego seleccionado

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
        )
        row = work.loc[selected_idx]

        st.markdown(f"### {row.get('name','(sin nombre)')}")
        meta_cols = [c for c in ["app_id", "release_date", "required_age", "price"] if c in row.index]
        st.json({c: row.get(c) for c in meta_cols})

        desc = row.get("detailed_description", "")
        if desc:
            st.markdown("**Descripción**")
            st.write(desc[:2000] + ("…" if len(desc) > 2000 else ""))

        if pd.notna(row.get("app_id", None)):
            st.markdown(f"**Steam:** https://store.steampowered.com/app/{int(row['app_id'])}/")

# -------------------------
# Recommender (TF-IDF)
# -------------------------
st.markdown("### 🤝 Recomendador simple (similares por descripción)")

if row is None:
    st.info("Selecciona un juego arriba para ver recomendaciones.")
    st.stop()

@st.cache_resource
def build_tfidf(texts: tuple, max_features: int = 20000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(list(texts))
    return vec, X

sample_n_rec = st.sidebar.slider("Tamaño muestra recomendador", 2000, 50000, 15000, step=1000)
top_k = st.sidebar.slider("Número de recomendaciones", 3, 20, 10)
max_feats = st.sidebar.slider("Max features TF-IDF", 5000, 50000, 20000, step=5000)

text_col = "detailed_description" if "detailed_description" in work.columns else "short_description"

base = work.copy()
if len(base) > sample_n_rec:
    base = base.sample(n=sample_n_rec, random_state=42)

base[text_col] = base[text_col].fillna("").astype(str)

if len(base) < 5 or base[text_col].str.len().sum() == 0:
    st.info("No hay texto suficiente para recomendar.")
    st.stop()

texts = tuple(base[text_col].tolist())
vec, X = build_tfidf(texts, max_features=max_feats)

# buscar el juego seleccionado dentro del subset
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
    st.info("El juego seleccionado no está en la muestra del recomendador. Sube la muestra o ajusta filtros.")
    st.stop()

i = base.index.get_loc(target_idx)
sims = cosine_similarity(X[i], X).flatten()
order = sims.argsort()[::-1]
order = [j for j in order if j != i][:top_k]

recs_cols = [c for c in ["app_id", "name", "price", "release_date", "genres"] if c in base.columns]
recs = base.iloc[order][recs_cols].copy()
st.dataframe(recs, use_container_width=True)