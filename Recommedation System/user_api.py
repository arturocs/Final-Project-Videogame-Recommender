import requests
import os
import re
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Cache global para resolver vanity URLs
vanity_cache = {}

def extract_steamid(entry):
    """
    Detecta automáticamente si el usuario ingresó:
    - SteamID numérico
    - URL con /profiles/
    - URL con /id/
    - Alias directo
    """

    if not entry:
        logging.error("Entrada vacía")
        return None

    entry = str(entry).strip()

    # 1️⃣ SteamID64 directo
    if entry.isdigit() and len(entry) == 17:
        logging.info("SteamID numérico detectado")
        return entry

    # 2️⃣ URL tipo /profiles/
    match_profile = re.search(r"/profiles/(\d+)", entry)
    if match_profile:
        logging.info("URL con SteamID detectada")
        return match_profile.group(1)

    # 3️⃣ URL tipo /id/alias
    match_vanity = re.search(r"/id/([^/?]+)", entry)
    if match_vanity:
        vanity = match_vanity.group(1).lower()
        if len(vanity) < 3:
            logging.error("Alias demasiado corto")
            return None
        logging.info(f"Alias detectado en URL: {vanity}")
        return resolve_vanity(vanity)

    # Alias directo
    if "/" not in entry:
        alias = entry.lower()
        if len(alias) < 3:
            logging.error("Alias demasiado corto")
            return None
        logging.info(f"Alias directo detectado: {alias}")
        return resolve_vanity(alias)

    logging.error("No se pudo interpretar la entrada")
    return None


def resolve_vanity(vanity):
    """
    Convierte un alias de Steam a SteamID64 usando la API.
    Usa cache para evitar llamadas repetidas.
    """
    vanity = vanity.lower()

    if vanity in vanity_cache:
        logging.info("Alias encontrado en cache")
        return vanity_cache[vanity]

    url = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
    params = {
        "key": API_KEY,
        "vanityurl": vanity
    }

    try:
        logging.info(f"Resolviendo alias: {vanity}")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logging.error(f"Error HTTP: {response.status_code}")
            return None

        data = response.json()
        if "response" not in data:
            logging.error("Respuesta inválida de Steam")
            return None

        if data["response"].get("success") == 1:
            steamid = data["response"].get("steamid")
            logging.info(f"Alias resuelto: {steamid}")
            vanity_cache[vanity] = steamid
            return steamid

        logging.error("Alias no encontrado o inválido")
        return None

    except requests.exceptions.Timeout:
        logging.error("Tiempo de espera agotado")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error de conexión: {e}")
        return None

    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return None


def get_user_games(steamid):
    """
    Consulta los juegos del usuario desde la API de Steam.
    """
    logging.info(f"Consultando juegos del usuario {steamid}")

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {
        "key": API_KEY,
        "steamid": steamid,
        "include_appinfo": True,
        "include_played_free_games": True
    }

    try:
        logging.info("Enviando petición a la API de Steam")
        response = requests.get(url, params=params, timeout=10)
        logging.info(f"Código de respuesta: {response.status_code}")

        if response.status_code != 200:
            logging.error("La API devolvió un error")
            return pd.DataFrame()

        data = response.json()
        games = data.get("response", {}).get("games", [])
        logging.info(f"Se encontraron {len(games)} juegos")

        return pd.DataFrame(games)

    except requests.exceptions.Timeout:
        logging.error("Tiempo de espera agotado al consultar juegos")
        return pd.DataFrame()

    except requests.RequestException as e:
        logging.error(f"Error en la petición: {e}")
        return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return pd.DataFrame()


def build_user_profile(df, user_games, min_minutes=120):
    """
    Construye un perfil del usuario filtrando juegos por tiempo mínimo jugado.
    """
    if user_games is None or user_games.empty:
        logging.warning("El usuario no tiene juegos o el perfil es privado")
        return df.iloc[0:0]

    # Filtrar juegos con al menos min_minutes
    filtered_games = user_games[user_games["playtime_forever"] >= min_minutes]
    logging.info(f"Juegos con al menos {min_minutes} minutos jugados: {len(filtered_games)}")

    if filtered_games.empty:
        logging.warning("Ningún juego supera el mínimo de tiempo jugado")
        return df.iloc[0:0]

    user_appids = set(filtered_games["appid"].astype(str))
    logging.info(f"AppIDs válidos: {len(user_appids)}")

    if "app_id" not in df.columns:
        logging.error("El DataFrame no contiene la columna 'app_id'")
        return df.iloc[0:0]

    # Filtrar dataset
    user_df = df[df["app_id"].isin(user_appids)]
    logging.info(f"Juegos encontrados en dataset: {len(user_df)}")

    return user_df