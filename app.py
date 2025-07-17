import os
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from steam_fetcher import fetch_and_save_steam_data
from recommend_core import load_and_process_data, compute_hybrid_scores, explain_recommendation,recommend_by_game_name_with_hybrid
from steam_processor import process_steam_data
from fuzzywuzzy import fuzz

st.set_page_config(page_title="Steam Game Recommender", layout="wide")
st.title("🎮 Steam Backlog Cleaner")

steam_id_input = st.text_input("Enter your SteamID64:", placeholder="76561197981871306")

def load_existing_data_and_model():
    try:
        if not os.path.exists("steam_all_processed.csv"):
            st.info("Processing steam_all.csv to generate processed data...")
            process_steam_data("steam_all.csv")
            st.success("Processing complete.")
        df, unplayed_df, embeddings_np, tag_index_map = load_and_process_data("steam_all_processed.csv")
        print("Columns in df:", df.columns)
        print("Columns in unplayed_df:", unplayed_df.columns)
        st.session_state['df'] = df
        st.session_state['unplayed_df'] = unplayed_df
        st.session_state['embeddings'] = embeddings_np
        st.session_state['tag_index_map'] = tag_index_map
        st.success("Loaded existing processed data and model.")
        return True
    except Exception as e:
        st.error(f"Failed loading existing data/model: {e}")
        return False

if os.path.exists("steam_all.csv"):
    st.info("Found existing steam_all.csv data, skipping fetching.")
    if 'df' not in st.session_state or 'embeddings' not in st.session_state:
        loaded = load_existing_data_and_model()
        if loaded:
            st.session_state['steam_all_df'] = pd.read_csv("steam_all.csv")
else:
    if st.button("Fetch My Steam Library"):
        if not steam_id_input.strip():
            st.error("Please enter a valid SteamID64.")
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            with st.spinner("Fetching Steam library and game data... This may take a few minutes."):
                try:
                    all_df, recent_df = fetch_and_save_steam_data(
                        steam_id_input.strip(),
                        progress_bar=progress_bar,
                        progress_text=progress_text,
                    )
                    if all_df is None or all_df.empty:
                        st.error("No games found for this SteamID or an error occurred.")
                    else:
                        st.success(f"Fetched {len(all_df)} owned games!")
                        st.session_state['steam_all_df'] = all_df
                        st.session_state['steam_recent_df'] = recent_df
                        st.session_state['steam_id'] = steam_id_input.strip()

                        all_df.to_csv("steam_all.csv", index=False)

                        st.info("Processing steam_all.csv...")
                        process_steam_data("steam_all.csv")
                        df, unplayed_df, embeddings_np, tag_index_map = load_and_process_data("steam_all_processed.csv")
                        st.session_state['df'] = df
                        st.session_state['unplayed_df'] = unplayed_df
                        st.session_state['embeddings'] = embeddings_np
                        st.session_state['tag_index_map'] = tag_index_map
                        st.success("Processing complete and model loaded.")

                    progress_bar.empty()
                    progress_text.empty()
                except Exception as e:
                    st.error(f"Failed to fetch data: {e}")
                    progress_bar.empty()
                    progress_text.empty()

# === Unified recommend function ===
def unified_recommend(
    input_name,
    top_n,
    df,
    unplayed_df,
    embeddings_np,
    tag_index_map,
    playtime_thresh=4,
    ach_thresh=5,
    play_ach_thresh=1
):
    if input_name:
        return recommend_by_game_name_with_hybrid(
            input_name, top_n, df, unplayed_df, embeddings_np, tag_index_map,
            playtime_thresh, ach_thresh, play_ach_thresh
        )
    else:
        results_df = compute_hybrid_scores(df, unplayed_df, embeddings_np, tag_index_map, top_n=top_n)
        results = [(row["name"], row["final_score"], row["new_cluster"]) for _, row in results_df.iterrows()]
        return results, None, df

# === Reason generation helper ===
def get_reasons_for_recommendation(game_name, df, liked_df, tag_index_map):
    row = df[df["name"] == game_name].iloc[0]
    explanation = explain_recommendation(row, liked_df, df, st.session_state['embeddings'], tag_index_map)
    reasons = [line.strip() for line in explanation.split("\n") if line.strip()]
    return reasons if reasons else ["Recommended based on your play history"]

# === UI for recommendations ===
if ('df' in st.session_state and 'embeddings' in st.session_state and
    'tag_index_map' in st.session_state and 'unplayed_df' in st.session_state):

    st.markdown("---")
    st.header("Get Game Recommendations")

    df = st.session_state['df']
    embeddings_np = st.session_state['embeddings']
    tag_index_map = st.session_state['tag_index_map']
    unplayed_df = st.session_state['unplayed_df']

    mode = st.radio("Choose recommendation mode:", ["Based on a game name", "Based on liked games"])
    selected_game = None

    def get_fuzzy_matches(name, df, limit=10):
        name = name.lower().strip()
        scored_titles = []
        for title in df["name"]:
            title_lower = title.lower()
            score_token_sort = fuzz.token_sort_ratio(title_lower, name)
            score_partial = fuzz.partial_ratio(title_lower, name)
            score_token_set = fuzz.token_set_ratio(title_lower, name)
            max_score = max(score_token_sort, score_partial, score_token_set)
            scored_titles.append((title, max_score))
        scored_titles.sort(key=lambda x: x[1], reverse=True)
        return [title for title, score in scored_titles[:limit]]


    if mode == "Based on a game name":
        typed_name = st.text_input("Enter a game name:", placeholder="e.g. Hollow Knight")
        if typed_name:
            matches = get_fuzzy_matches(typed_name, df)
            if matches:
                selected_game = st.selectbox("Did you mean:", options=matches)
            else:
                st.warning("No close matches found. Try typing another name.")
    else:
        selected_game = None

    top_n = st.slider("Number of recommendations:", 5, 50, 10)

    with st.expander("⚙️ Advanced Settings", expanded=False):
        play_thresh = st.slider("Min playtime for a game to be liked (hrs):", 0, 10, 4)
        ach_thresh = st.slider("Min achievement % for liked:", 0, 100, 5)
        play_ach_thresh = st.slider("Min playtime for achievement consideration (hrs):", 0, 10, 10)

    card_css = """
    <style>
    .card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.12);
        background-color: #fff;
        transition: box-shadow 0.3s ease-in-out;
    }
    .card:hover {
        box-shadow: 5px 5px 15px rgba(0,0,0,0.25);
    }
    .card img {
        border-radius: 8px;
        max-width: 100%;
        height: auto;
    }
    .tag-list {
        font-style: italic;
    }
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    import ast

    def safe_eval_tag_weights(x):
        if isinstance(x, dict):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}

    df['tag_weights'] = df['tag_weights'].apply(safe_eval_tag_weights)
    unplayed_df['tag_weights'] = unplayed_df['tag_weights'].apply(safe_eval_tag_weights)

    def display_game_recommendation(game, reasons, score):
        tag_weights = game['tag_weights']
        if not isinstance(tag_weights, dict):
            tag_weights = {}
        top_tags = sorted(tag_weights, key=tag_weights.get, reverse=True)[:7]

        cols = st.columns([3, 5])
        with cols[0]:
            st.image(game['image_url'], use_container_width=True)

        with cols[1]:
            steam_url = f"https://store.steampowered.com/app/{game['appid']}"
            game_name_link = f"[{game['name']}]({steam_url})"
            st.markdown(f"### 🎮 {game_name_link}", unsafe_allow_html=True)
            st.markdown(f"**Tags:** {', '.join(top_tags)}")

            release_year = int(game['release_year']) if pd.notna(game['release_year']) else "Unknown"
            price_str = f"${game['price_usd']:.2f}"
            achievement_pct = game['achievement_pct']
            achievement_str = f"{achievement_pct:.1f}%" if pd.notna(achievement_pct) else "N/A"
            playtime_str = f"{game['playtime_hours']:.1f} hrs"
            dev_emoji = "🏢"

            row1 = f"{dev_emoji} **Developer:** {game['first_developer']} &nbsp;&nbsp;&nbsp; 📅 **Year:** {release_year} &nbsp;&nbsp;&nbsp; 💰 **Price:** {price_str}"
            st.markdown(row1, unsafe_allow_html=True)

            row2 = f"🕒 **Playtime:** {playtime_str} &nbsp;&nbsp;&nbsp; 🏆 **Achievements:** {achievement_str}"
            st.markdown(row2, unsafe_allow_html=True)

            st.markdown("**Why recommended:**")

            dev_prefix = "• "
            dev_lines = [r for r in reasons if r.startswith(dev_prefix)]
            other_lines = [r for r in reasons if not r.startswith(dev_prefix)]

            max_dev_games = 4
            if len(dev_lines) > max_dev_games:
                shown_dev_lines = dev_lines[:max_dev_games]
                shown_dev_lines.append("• ...")
            else:
                shown_dev_lines = dev_lines

            for line in other_lines:
                if line.startswith("Because it's similar to:"):
                    st.markdown(f"- {line}")
                else:
                    st.text(f"- {line}")

            for line in shown_dev_lines:
                st.text(line)

            st.markdown("---")


    if st.button("🔍 Recommend Games"):
        results, target, df = unified_recommend(
            input_name=selected_game if mode == "Based on a game name" else None,
            top_n=top_n,
            df=df,
            unplayed_df=unplayed_df,
            embeddings_np=embeddings_np,
            tag_index_map=tag_index_map,
            playtime_thresh=play_thresh,
            ach_thresh=ach_thresh,
            play_ach_thresh=play_ach_thresh,
        )


        liked_games = df[df["liked"] == True].copy()

        if results:
            for name, score, cluster in results:
                game = df[df["name"] == name].iloc[0]
                reasons = get_reasons_for_recommendation(name, df, liked_games, tag_index_map)
                display_game_recommendation(game, reasons, score)
        else:
            st.warning("No recommendations found.")

else:
    st.info("Enter your SteamID64 above and fetch your library to unlock recommendations.")
