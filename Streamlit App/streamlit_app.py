# -------------------------
# Setting Libraries
# -------------------------
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    st.subheader("🔮 Prompt Bot (recomendación por texto/mood)")
    st.caption("Primero aplicamos un filtro compuesto, luego usamos TF-IDF para recomendar dentro del subset filtrado.")

    prompt = st.text_area(
        "Tu prompt",
        placeholder="Ej: Estoy cansada y quiero algo cozy, relajante, sin terror, que pueda jugar sola.",
        height=90,
        key="prompt_text",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        top_k_prompt = st.slider("Top K", 3, 30, 10, key="topk_prompt")
    with c2:
        sample_n_prompt = st.slider("Tamaño muestra", 2000, 50000, 15000, step=1000, key="sample_prompt")
    with c3:
        max_feats_prompt = st.slider("Max features TF-IDF", 5000, 50000, 20000, step=5000, key="max_feats_prompt")

    use_extra = st.checkbox(
        "Usar géneros/tags además de descripción",
        value=True,
        key="use_extra_prompt"
    )

    text_col = pick_text_col(work)
    if not text_col:
        st.warning("No hay columna de texto (short/detailed_description). El prompt bot necesita texto.")
    else:
        advanced_enabled = st.toggle("Activar filtro avanzado", value=False, key="advanced_enabled")

        # Detectar columnas reales del dataset
        genre_col = first_existing_col(work, ["genres", "genre"])
        category_col = first_existing_col(work, ["categories", "category"])
        tag_col = first_existing_col(work, ["tags", "tag"])
        platform_col = first_existing_col(work, ["platforms", "platform"])
        developer_col = first_existing_col(work, ["developers", "developer"])
        publisher_col = first_existing_col(work, ["publishers", "publisher"])

        with st.expander("🧩 Filtro compuesto (avanzado)", expanded=advanced_enabled):

            selected_genres = []
            selected_categories = []
            selected_tags = []
            selected_platforms = []
            selected_developers = []
            selected_publishers = []

            if genre_col:
                genre_options = build_options_from_col(work, genre_col, top_n=150)
                selected_genres = st.multiselect(
                    "🎭 Géneros",
                    genre_options,
                    default=[],
                    key="selected_genres",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de géneros en el dataset.")

            if category_col:
                cat_options = build_options_from_col(work, category_col, top_n=150)
                selected_categories = st.multiselect(
                    "🧩 Categorías",
                    cat_options,
                    default=[],
                    key="selected_categories",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de categorías en el dataset.")

            if tag_col:
                tag_options = build_options_from_col(work, tag_col, top_n=150)
                selected_tags = st.multiselect(
                    "🏷️ Tags",
                    tag_options,
                    default=[],
                    key="selected_tags",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de tags en el dataset.")

            if platform_col:
                platform_options = build_options_from_col(work, platform_col, top_n=100)
                selected_platforms = st.multiselect(
                    "🕹️ Plataformas",
                    platform_options,
                    default=[],
                    key="selected_platforms",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de plataformas en el dataset.")

            if developer_col:
                developer_options = build_options_from_col(work, developer_col, top_n=100)
                selected_developers = st.multiselect(
                    "👨‍💻 Developers",
                    developer_options,
                    default=[],
                    key="selected_developers",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de developers en el dataset.")

            if publisher_col:
                publisher_options = build_options_from_col(work, publisher_col, top_n=100)
                selected_publishers = st.multiselect(
                    "🏢 Publishers",
                    publisher_options,
                    default=[],
                    key="selected_publishers",
                    disabled=not advanced_enabled
                )
            else:
                st.caption("No hay columna de publishers en el dataset.")

            include_kw = st.text_input(
                "Incluir palabras (coma)",
                "",
                key="include_kw",
                disabled=not advanced_enabled
            )

            exclude_kw = st.text_input(
                "Excluir palabras (coma)",
                "",
                key="exclude_kw",
                disabled=not advanced_enabled
            )

        filtered = apply_composite_filter(
            df_in=work,
            genre_col=genre_col,
            category_col=category_col,
            tag_col=tag_col,
            platform_col=platform_col,
            developer_col=developer_col,
            publisher_col=publisher_col,
            selected_genres=selected_genres,
            selected_categories=selected_categories,
            selected_tags=selected_tags,
            selected_platforms=selected_platforms,
            selected_developers=selected_developers,
            selected_publishers=selected_publishers,
            include_kw=include_kw,
            exclude_kw=exclude_kw,
            text_col_for_kw=text_col
        )

        st.write(f"Después del filtro compuesto: **{len(filtered):,}** juegos")

        if st.button("Recomiéndame por prompt", key="btn_prompt"):
            if not prompt.strip():
                st.warning("Escribe un prompt primero.")
            elif len(filtered) < 5:
                st.warning("Tras filtros quedan muy pocos juegos. Afloja filtros o aumenta dataset.")
            else:
                base2 = filtered.copy()
                if len(base2) > sample_n_prompt:
                    base2 = base2.sample(n=sample_n_prompt, random_state=42)

                extra_cols = []
                if use_extra:
                    for c in [genre_col, category_col, tag_col, platform_col, developer_col, publisher_col]:
                        if c is not None and c in base2.columns:
                            extra_cols.append(c)

                base2[text_col] = base2[text_col].fillna("").astype(str)
                for c in extra_cols:
                    base2[c] = base2[c].fillna("").astype(str)

                def make_doc(r):
                    parts = [safe_str(r.get(text_col, ""))]
                    for c in extra_cols:
                        parts.append(safe_str(r.get(c, "")))
                    return " ".join(parts)

                docs = tuple(base2.apply(make_doc, axis=1).tolist())
                vec2, X2 = build_tfidf_matrix(docs, max_features=max_feats_prompt)

                qv = vec2.transform([prompt])
                sims2 = cosine_similarity(qv, X2).flatten()
                order2 = sims2.argsort()[::-1][:top_k_prompt]

                recs_cols2 = [
                    c for c in [
                        "app_id", "name", "price", "release_date",
                        genre_col, category_col, tag_col
                    ]
                    if c is not None and c in base2.columns
                ]
                out = base2.iloc[order2][recs_cols2].copy()
                out["similarity"] = sims2[order2]
                st.dataframe(out, use_container_width=True)


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

    steam_id = st.text_input(
        "Steam ID del usuario, alias o url de perfil",
        placeholder="Ej: 7656119XXXXXXXXXX",
        key="steam_user_id"
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        load_user_btn = st.button("Buscar usuario", key="load_user_btn")
    with c2:
        top_k_user = st.slider("Top recomendaciones", 3, 20, 10, key="top_k_user")

    # Sitio donde guardamos el perfil cargado
    if "loaded_user_profile" not in st.session_state:
        st.session_state["loaded_user_profile"] = None

    if load_user_btn:
        if not steam_id.strip():
            st.warning("Introduce un Steam ID.")
        else:
            with st.spinner("Buscando usuario..."):
                # TODO: sustituir por API real
                profile = fetch_user_profile_mock(steam_id)

            if not profile.get("ok", False):
                st.error("No se pudo cargar el usuario. Puede que el perfil sea privado o el ID no sea válido.")
                st.session_state["loaded_user_profile"] = None
            else:
                st.session_state["loaded_user_profile"] = profile
                st.success("Usuario cargado correctamente ✅")

    profile = st.session_state.get("loaded_user_profile")

    if profile:
        st.markdown("### 👤 Perfil del usuario")
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(f"*Usuario:* {profile.get('user_name', 'N/A')}")
            st.write(f"*Steam ID:* {profile.get('steam_id', 'N/A')}")
            st.write(f"*Perfil público:* {'Sí' if profile.get('is_public') else 'No'}")

        with info_col2:
            st.write("*Géneros favoritos:*")
            fav_genres = profile.get("favorite_genres", [])
            if fav_genres:
                st.write(", ".join(fav_genres))
            else:
                st.write("No disponible")

            st.write("*Tags favoritas:*")
            fav_tags = profile.get("favorite_tags", [])
            if fav_tags:
                st.write(", ".join(fav_tags))
            else:
                st.write("No disponible")

        st.markdown("### 🎮 Juegos más jugados")
        top_games = profile.get("top_games", [])
        if top_games:
            st.dataframe(pd.DataFrame(top_games), use_container_width=True)
        else:
            st.info("No hay juegos para mostrar.")

        st.divider()

        if st.button("Recomiéndame en base a este usuario", key="recommend_user_btn"):
            with st.spinner("Generando recomendaciones..."):
                # TODO: sustituir por API real
                recs_user = fetch_user_recommendations_mock(profile["steam_id"], top_k=top_k_user)

            st.markdown("### ✨ Recomendaciones para este usuario")
            if recs_user is not None and len(recs_user) > 0:
                st.dataframe(recs_user, use_container_width=True)
            else:
                st.warning("No se encontraron recomendaciones.")
    else:
        st.info("Carga un usuario para ver su perfil y generar recomendaciones.")