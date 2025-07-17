import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()  

API_KEY = os.getenv("STEAM_API_KEY")

OWNED_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
ACH_URL = "https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/"
STORE_URL = "https://store.steampowered.com/api/appdetails"

def fetch_owned(steam_id):
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "include_appinfo": 1,
        "include_played_free_games": 1,
        "format": "json"
    }
    r = requests.get(OWNED_URL, params=params)
    r.raise_for_status()
    return r.json().get("response", {}).get("games", [])

def fetch_ach(appid, steam_id):
    try:
        r = requests.get(ACH_URL, params={
            "key": API_KEY,
            "steamid": steam_id,
            "appid": appid
        })
        r.raise_for_status()
        data = r.json().get("playerstats", {})
        achs = data.get("achievements", [])
        if achs:
            achieved_count = sum(a["achieved"] for a in achs)
            return round(achieved_count / len(achs) * 100, 2)
    except:
        pass
    return None

def fetch_store(appid):
    try:
        r = requests.get(STORE_URL, params={"appids": appid, "cc": "us", "l": "en"})
        r.raise_for_status()
        data = r.json().get(str(appid), {}).get("data", {})
        if not data or data.get("type") != "game":
            return {}

        price_info = data.get("price_overview", {})
        if data.get("is_free", False):
            price = "Free"
        elif price_info:
            price = round(price_info.get("initial", 0) / 100, 2)
        else:
            price = None

        return {
            "genres": ", ".join(g["description"] for g in data.get("genres", [])),
            "metacritic": data.get("metacritic", {}).get("score"),
            "developer": "; ".join(data.get("developers", [])) or None,
            "publisher": "; ".join(data.get("publishers", [])) or None,
            "price": price,
            "release_date": data.get("release_date", {}).get("date"),
            "image": data.get("header_image")
        }
    except:
        return {}

def fetch_spy_tags_and_score(appid):
    try:
        url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        tags = list(data.get("tags", {}).keys())
        pos = data.get("positive", 0)
        neg = data.get("negative", 0)
        score_pct = round((pos / (pos + neg)) * 100, 2) if (pos + neg) > 0 else None
        return tags, score_pct
    except:
        return [], None

def collect_data(games, steam_id, progress_bar=None, progress_text=None):
    rows = []
    total = len(games)
    for i, g in enumerate(games):
        appid = g["appid"]
        name = g.get("name", "")
        playtime = round(g.get("playtime_forever", 0) / 60, 2)  
        ach = fetch_ach(appid, steam_id)
        store = fetch_store(appid)
        tags, user_score = fetch_spy_tags_and_score(appid)

        developer = store.get("developer")
        publisher = store.get("publisher")
        if not developer and publisher:
            developer = publisher
        if not publisher and developer:
            publisher = developer

        rows.append({
            "appid": appid,
            "name": name,
            "playtime_hours": playtime,
            "achievement_pct": ach,
            "genres": store.get("genres"),
            "tags": ", ".join(tags),
            "metacritic_score": store.get("metacritic"),
            "user_score_pct": user_score,
            "developer": developer,
            "publisher": publisher,
            "price_usd": store.get("price"),
            "release_date": store.get("release_date"),
            "image_url": store.get("image")
        })

        if progress_bar is not None and progress_text is not None:
            progress_bar.progress((i + 1) / total)
            progress_text.text(f"Fetched {i + 1}/{total} games: {name}")

        sleep(1)  

    return pd.DataFrame(rows)

def fetch_and_save_steam_data(steam_id, progress_bar=None, progress_text=None):
    owned_games = fetch_owned(steam_id)
    if not owned_games:
        return None, None

    all_df = collect_data(owned_games, steam_id, progress_bar, progress_text)
    recent_df = collect_data([g for g in owned_games if g.get("playtime_2weeks", 0) > 0], steam_id, progress_bar, progress_text)

    all_df.to_csv("steam_all.csv", index=False)
    recent_df.to_csv("steam_recent.csv", index=False)
    return all_df, recent_df
