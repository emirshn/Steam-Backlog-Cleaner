import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Constants
BLOCKED_TAGS = {"hentai", "nudity", "sexual content", "nsfw", "mature", "eroge", "adult", "poarn"}
TAG_STOPLIST = {
    "female protagonist", "soundtrack", "memes", "anime", "narrative", "controller",
    "great soundtrack", "early access", "free to play",
    "multiplayer", "singleplayer", "co-op", "local multiplayer", "online multiplayer",
    "indie", "action", "adventure", "rpg", "fps", "strategy", "simulation",
    "casual", "pixel graphics", "story rich", "open world",
    "family friendly", "mature", "violent", "gore",
    "cute", "cartoon", "fantasy", "sci-fi", "horror", "survival",
    "sandbox", "2d", "3d", "controller support", "steam achievements",
    "steam cloud", "steam workshop", "vr", "early access",
    "tutorial", "moddable", "single-player", "online co-op"
}
DEV_SUFFIXES = ["ltd", "inc", "llc", "co", "corp", "corporation", "limited", "entertainment", "studios", "studio", "game", "games"]

# Helpers
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

def normalize_dev_name(dev_name: str) -> str:
    dev = dev_name.lower()
    for suffix in DEV_SUFFIXES:
        dev = re.sub(r'\b' + re.escape(suffix) + r'\b\.?', '', dev)
    dev = re.sub(r'[^\w\s]', '', dev)
    return dev.strip()

def normalize_developers(dev_list):
    return [normalize_dev_name(d).title() for d in dev_list if d.strip()]

def is_explicit_game(row) -> bool:
    tags = {t.lower() for t in row["tags"]}
    return len(tags & BLOCKED_TAGS) >= 2

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def load_and_process_data(csv_path, liked_ach_thresh=5, unplayed_ach_thresh=5, min_playtime=4):
    df = pd.read_csv(csv_path)

    # Parse list fields safely
    for col in ["tags", "genres", "developer", "publisher"]:
        if col in df.columns:
            df[col] = df[col].fillna("[]").apply(safe_literal_eval)

    # Normalize developer names
    df["developer"] = df["developer"].apply(normalize_developers)

    # Filter explicit and cheap games
    df = df[~df.apply(is_explicit_game, axis=1)]
    df = df[df["price_usd_num"] >= 5]

    # Mark liked/unplayed games
    df["liked"] = (
        (df["achievement_pct_num"] >= liked_ach_thresh) |
        ((df["achievement_pct_num"] == 0) & (df["playtime_hours"] >= min_playtime)) |
        ((df["achievement_pct_num"] < liked_ach_thresh) &
         (df["achievement_pct_num"] > 0) &
         (df["playtime_hours"] >= min_playtime))
    )

    unplayed_df = df[
        (df["achievement_pct_num"] < unplayed_ach_thresh) &
        (df["price_usd_num"] > 0) &
        (df["playtime_hours"] <= min_playtime)
    ].copy()

    # MultiLabelBinarizer for tags (binarized)
    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df["tags"])

    # Normalize numeric columns to [0,1] using MinMax scaling 
    numeric_cols = ["metacritic_score_num", "user_score_pct_num", "price_usd_num", "achievement_pct_num", "playtime_hours"]
    numeric_data = df[numeric_cols].fillna(0)
    numeric_min = numeric_data.min()
    numeric_max = numeric_data.max()
    numeric_scaled = (numeric_data - numeric_min) / (numeric_max - numeric_min + 1e-9)

    # Combine binary tags and normalized numeric data
    X = np.hstack([tags_encoded, numeric_scaled])

    model = Autoencoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    liked_mask = df["liked"].values
    liked_X = X[liked_mask]
    dataset = TensorDataset(torch.tensor(liked_X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(15): 
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0]
            decoded, _ = model(x)
            loss = loss_fn(decoded, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(loader.dataset)

    # Get embeddings for all games
    with torch.no_grad():
        _, embeddings = model(torch.tensor(X, dtype=torch.float32))
    embeddings_np = embeddings.numpy()

    return df, unplayed_df, embeddings_np, mlb.classes_

def compute_hybrid_scores(df, unplayed_df, embeddings_np, tag_classes, top_n=10):
    liked_df = df[df["liked"]]
    
    if liked_df.empty:
        return pd.DataFrame()

    # Embedding similarity 
    liked_mask = df["liked"].values
    liked_emb_mean = embeddings_np[liked_mask].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(embeddings_np, liked_emb_mean).flatten()
    df["sim_score"] = sims

    #  Merge required columns into unplayed_df ---
    unplayed_df = unplayed_df.merge(
        df[["appid", "sim_score", "tags", "developer", "new_cluster"]],
        on="appid",
        how="left",
        suffixes=("", "_df")
    )

    #  Cluster bonus 
    liked_clusters = set(liked_df["new_cluster"])
    unplayed_df["cluster_bonus"] = unplayed_df["new_cluster"].apply(
        lambda c: 1.0 if c in liked_clusters else 0.0
    )

    # Tag score (based on liked tag frequency) 
    liked_tags = Counter(
        tag.lower() for tags in liked_df["tags"] for tag in tags if tag.lower() not in TAG_STOPLIST
    )
    max_tag_play = max(liked_tags.values()) if liked_tags else 1

    def tag_score(tags):
        return sum(liked_tags.get(t.lower(), 0) for t in tags) / max_tag_play

    unplayed_df["tag_score"] = unplayed_df["tags"].apply(tag_score)

    # Developer score (based on total playtime on same devs) 
    dev_playtime = Counter()
    for _, row in liked_df.iterrows():
        for dev in row["developer"]:
            dev_playtime[dev] += row["playtime_hours"]
    max_dev_play = max(dev_playtime.values()) if dev_playtime else 1

    def dev_score(devs):
        return sum(dev_playtime.get(d, 0) for d in devs) / max_dev_play

    unplayed_df["dev_score"] = unplayed_df["developer"].apply(dev_score)

    #  Normalize embedding similarity 
    min_sim, max_sim = unplayed_df["sim_score"].min(), unplayed_df["sim_score"].max()
    unplayed_df["sim_norm"] = (unplayed_df["sim_score"] - min_sim) / (max_sim - min_sim + 1e-9)

    #  Final weighted score including cluster bonus 
    w_emb, w_tag, w_dev, w_cluster = 0.55, 0.30, 0.1, 0.05
    unplayed_df["final_score"] = (
        w_emb * unplayed_df["sim_norm"] +
        w_tag * unplayed_df["tag_score"] +
        w_dev * unplayed_df["dev_score"] +
        w_cluster * unplayed_df["cluster_bonus"]
    )

    return unplayed_df.sort_values("final_score", ascending=False).head(top_n)

def recommend_by_game_name_with_hybrid(
    input_name,
    top_n,
    df,
    unplayed_df,
    embeddings_np,
    playtime_thresh=4,
    ach_thresh=5,
    play_ach_thresh=1
):
    # Recompute liked mask
    df["liked"] = (
        (df["achievement_pct_num"] >= ach_thresh) |
        ((df["achievement_pct_num"] == 0) & (df["playtime_hours"] >= play_ach_thresh)) |
        ((df["achievement_pct_num"] < ach_thresh) &
         (df["achievement_pct_num"] > 0) & (df["playtime_hours"] >= play_ach_thresh))
    )

    # Recompute unplayed_df (safe fallback if stale)
    unplayed_df = df[
        (df["achievement_pct_num"] < ach_thresh) &
        (df["price_usd_num"] > 0) &
        (df["playtime_hours"] <= playtime_thresh)
    ].copy()

    # Find target game
    target_game = df[df["name"] == input_name]
    if target_game.empty:
        return [], None, df
    target_row = target_game.iloc[0]
    target_idx = df.index.get_loc(target_row.name)
    target_embedding = embeddings_np[target_idx].reshape(1, -1)

    # Compute embedding similarity for unplayed games
    unplayed_indices = [df.index.get_loc(i) for i in unplayed_df.index]
    unplayed_embeddings = embeddings_np[unplayed_indices]
    sims = cosine_similarity(unplayed_embeddings, target_embedding).flatten()
    unplayed_df["sim_score"] = sims

    # --- Tag affinity: based on shared tags with target game ---
    target_tags = set([tag.lower() for tag in target_row["tags"] if tag.lower() not in TAG_STOPLIST])
    def tag_score(tags):
        return len(target_tags & set([t.lower() for t in tags])) / (len(target_tags) + 1e-9)
    unplayed_df["tag_score"] = unplayed_df["tags"].apply(tag_score)

    # --- Developer affinity ---
    target_devs = set(target_row["developer"])
    def dev_score(devs):
        return len(target_devs & set(devs)) / (len(target_devs) + 1e-9)
    unplayed_df["dev_score"] = unplayed_df["developer"].apply(dev_score)

    # --- Cluster bonus ---
    target_cluster = target_row["new_cluster"]
    unplayed_df["cluster_bonus"] = unplayed_df["new_cluster"].apply(lambda c: 1.0 if c == target_cluster else 0.0)

    # --- Normalize sim_score ---
    min_sim, max_sim = unplayed_df["sim_score"].min(), unplayed_df["sim_score"].max()
    unplayed_df["sim_norm"] = (unplayed_df["sim_score"] - min_sim) / (max_sim - min_sim + 1e-9)

    # --- Final score ---
    w_emb, w_tag, w_dev, w_cluster = 0.55, 0.30, 0.10, 0.05
    unplayed_df["final_score"] = (
        w_emb * unplayed_df["sim_norm"] +
        w_tag * unplayed_df["tag_score"] +
        w_dev * unplayed_df["dev_score"] +
        w_cluster * unplayed_df["cluster_bonus"]
    )

    top_recommendations = unplayed_df.sort_values("final_score", ascending=False).head(top_n)
    results = [(row["name"], row["final_score"], row["new_cluster"]) for _, row in top_recommendations.iterrows()]
    return results, target_row, df

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def explain_recommendation(row, liked_df, df, embeddings_np, top_n=3):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Sanitize top_n
    try:
        top_n = int(top_n)
        if top_n < 1:
            top_n = 3
    except Exception:
        top_n = 3

    # Find positional index of the game in df to align with embeddings
    try:
        game_pos_idx = df.index.get_loc(df.index[df["appid"] == row["appid"]][0])
    except Exception:
        return "❌ Could not find game in dataset."

    game_vec = embeddings_np[game_pos_idx].reshape(1, -1)

    # Get cluster of the recommended game from new_cluster
    game_cluster = df.loc[df["appid"] == row["appid"], "new_cluster"].values[0]

    # Filter liked games that are in the same new_cluster
    liked_same_cluster = liked_df[liked_df["new_cluster"] == game_cluster]

    # Initialize similar_game_texts as empty list always
    similar_game_texts = []

    if liked_same_cluster.empty:
        return "❌ No liked games found in the same cluster for similarity comparison."

    # Get positional indices of liked games in the cluster
    liked_positions = []
    for idx in liked_same_cluster.index:
        try:
            liked_positions.append(df.index.get_loc(idx))
        except KeyError:
            continue  # skip if index not found

    if not liked_positions:
        return "❌ No liked games found in the same cluster with embeddings."

    liked_vectors = embeddings_np[liked_positions]

    # Compute cosine similarities
    similarities = cosine_similarity(game_vec, liked_vectors).flatten()

    actual_top_n = min(top_n, len(similarities))
    top_indices = similarities.argsort()[-actual_top_n:][::-1]

    # Select the top similar liked games
    top_similars = liked_same_cluster.iloc[top_indices]

    # Format similar games info
    for _, game in top_similars.iterrows():
        ach_str = f"{game.get('achievement_pct_num', 0):.0f}%" if game.get('achievement_pct_num', 0) > 0 else "no achievements"
        similar_game_texts.append(f"{game['name']} ({game.get('playtime_hours', 0):.1f}h, {ach_str})")

    # Helper to union sets safely
    def union_of_sets(iterable):
        sets = [set(x) if isinstance(x, (list, set)) else set() for x in iterable if x]
        return set.union(*sets) if sets else set()

    shared_tags = set(row["tags"]) & union_of_sets(top_similars["tags"])
    shared_genres = set(row["genres"]) & union_of_sets(top_similars["genres"])

    # developer might be string or list, normalize to set
    def to_set(dev):
        if isinstance(dev, str):
            return {dev}
        elif isinstance(dev, (list, set)):
            return set(dev)
        return set()

    shared_devs = to_set(row["developer"]) & union_of_sets(top_similars["developer"].apply(to_set))

    # Build explanation parts
    parts = []
    if similar_game_texts:
        parts.append("Because it's similar to: " + ", ".join(similar_game_texts))
    if shared_tags:
        parts.append(f"🔖 Shared tags: {', '.join(sorted(shared_tags)[:5])}")
    if shared_genres:
        parts.append(f"🎮 Shared genres: {', '.join(sorted(shared_genres)[:3])}")
    if shared_devs:
        liked_same_dev_games = liked_df[
            liked_df['developer'].apply(to_set).apply(lambda devs: bool(devs & to_set(row['developer'])))
        ]
        dev_games_texts = []
        for _, g in liked_same_dev_games.iterrows():
            achievement_pct = g.get("achievement_pct_num", 0)
            ach_str = f"{achievement_pct:.0f}%" if achievement_pct > 0 else "no achievements"
            dev_games_texts.append(f"{g.get('name')} ({g.get('playtime_hours', 0):.1f}h, {ach_str})")
        parts.append("🧪 Liked games from the same developer(s): " + ", ".join(dev_games_texts))

    return "\n".join(parts)
