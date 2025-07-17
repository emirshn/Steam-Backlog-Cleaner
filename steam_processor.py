import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import umap
import plotly.express as px
import plotly.io as pio

# --- CONFIG ---
MIN_PRICE = 5.0

DEV_SUFFIXES = {
    "ltd", "inc", "llc", "co", "corp", "corporation",
    "limited", "entertainment", "studios", "studio", "game", "games"
}

def process_steam_data(CSV_PATH):
    # --- Helpers ---
    def ensure_list(x, sep=", "):
        if isinstance(x, list): return x
        if isinstance(x, str): return x.split(sep) if x else []
        return []

    def normalize_name_simple(name):
        if not isinstance(name, str): return ""
        name = name.lower()
        for suffix in DEV_SUFFIXES:
            name = re.sub(rf'\b{re.escape(suffix)}\.?\b', '', name)
        name = re.sub(r'[^\w\s]', '', name)
        return re.sub(r'\s+', ' ', name).strip().title()

    def fix_dev_with_pub(dev_list, pub_list):
        if isinstance(dev_list, list) and len(dev_list) > 1 and isinstance(pub_list, list) and pub_list:
            return [normalize_name_simple(pub_list[0])]
        return [normalize_name_simple(dev) for dev in dev_list]

    def weight_tags_linear(tags):
        n = len(tags)
        return {tag: 1 - i / n for i, tag in enumerate(tags)} if n else {}

    def normalize_weights(d):
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s else d

    def weighted_jaccard(d1, d2):
        keys = d1.keys() | d2.keys()
        shared = sum(min(d1.get(k, 0), d2.get(k, 0)) for k in keys)
        union = sum(max(d1.get(k, 0), d2.get(k, 0)) for k in keys)
        return shared / union if union else 0

    # --- Load & preprocess ---
    df = pd.read_csv(CSV_PATH).fillna("")
    df_raw = df.copy()

    for col, sep in [("tags", ", "), ("genres", ", "), ("developer", ";"), ("publisher", ";")]:
        df[col] = df[col].apply(lambda x: ensure_list(x, sep=sep))

    df["developer"] = [fix_dev_with_pub(devs, pubs) for devs, pubs in zip(df["developer"], df["publisher"])]
    df["first_developer"] = df["developer"].apply(lambda x: x[0] if x else "")
    df["developer"] = df["first_developer"].apply(lambda x: [x] if x else [])
    df["first_publisher"] = df["publisher"].apply(lambda x: x[0] if x else "")
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    df["tag_weights"] = df["tags"].apply(weight_tags_linear).apply(normalize_weights)

    for col in ["achievement_pct", "playtime_hours", "price_usd", "user_score_pct", "metacritic_score"]:
        df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- NSFW Filtering ---
    blocked_tags = {"hentai", "nudity", "sexual content", "nsfw", "mature", "eroge", "adult", "porn"}
    def is_explicit_game(row):
        tags = set(t.lower() for t in row["tags"])
        return len(tags & blocked_tags) >= 2 and row["metacritic_score_num"] < 80

    df = df[~df.apply(is_explicit_game, axis=1)]
    df = df[df["price_usd_num"] >= MIN_PRICE]

    # --- Vectorize ---
    vectorizer = DictVectorizer(sparse=False)
    tag_weight_matrix = vectorizer.fit_transform(df["tag_weights"])

    # --- UMAP + DBSCAN Clustering ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    df = df.reset_index(drop=True)
    embedding = reducer.fit_transform(tag_weight_matrix)

    dbscan = DBSCAN(eps=0.3, min_samples=5)
    df["cluster"] = dbscan.fit_predict(embedding)

    # --- Recursive Subclustering ---
    def automate_subclustering_recursive(df, embedding, cluster_col='cluster',
                                        eps_sub=0.1, min_samples_sub=3,
                                        cluster_size_threshold=50, max_iter=10):
        labels = df[cluster_col].to_numpy().copy()
        max_label = labels.max()
        for i in range(max_iter):
            big_clusters = [cl for cl in np.unique(labels) if cl != -1 and np.sum(labels == cl) > cluster_size_threshold]
            if not big_clusters:
                break
            updated = False
            for cl in big_clusters:
                idxs = np.where(labels == cl)[0]
                sub_emb = embedding[idxs]
                sub_labels = DBSCAN(eps=eps_sub, min_samples=min_samples_sub).fit_predict(sub_emb)
                valid_sub = set(sub_labels)
                if len(valid_sub) <= 1 and (-1 in valid_sub or len(valid_sub) == 1):
                    num_subclusters = int(np.ceil(len(idxs) / cluster_size_threshold))
                    if num_subclusters > 1:
                        kmeans = KMeans(n_clusters=num_subclusters, random_state=42)
                        km_labels = kmeans.fit_predict(sub_emb)
                        mapping = {}
                        for lbl in np.unique(km_labels):
                            max_label += 1
                            mapping[lbl] = max_label
                        for idx, lbl in zip(idxs, km_labels):
                            labels[idx] = mapping[lbl]
                        updated = True
                else:
                    mapping = {}
                    for lbl in valid_sub:
                        if lbl == -1: continue
                        max_label += 1
                        mapping[lbl] = max_label
                    for idx, lbl in zip(idxs, sub_labels):
                        if lbl != -1:
                            labels[idx] = mapping[lbl]
                    updated = True
            if not updated:
                break
        return labels

    df['new_cluster'] = automate_subclustering_recursive(df, embedding, cluster_col='cluster', eps_sub=0.2, min_samples_sub=5, cluster_size_threshold=50)

    # --- Genre Profiles ---
    genre_examples = {
        "CRPG": [
            "Baldur's Gate", "Planescape: Torment", "Pillars of Eternity",
            "Divinity: Original Sin 2", "Neverwinter Nights", "Wasteland 3",
            "Icewind Dale", "Fallout: New Vegas", "Tyranny", "Pathfinder: Kingmaker",
            "Disco Elysium", "Encased", "Arcanum: Of Steamworks and Magick Obscura"
        ],
        "JRPG": [
            "Persona 5", "Chrono Trigger", "Final Fantasy X",
            "Tales of Symphonia", "Ys VIII", "The Legend of Heroes: Trails of Cold Steel",
            "Ni no Kuni: Wrath of the White Witch", "Shin Megami Tensei III Nocturne",
            "Bravely Default II", "Dragon Quest XI", "Blue Reflection", "Valkyria Chronicles"
        ],
        "ARPG": [
            "Diablo III", "Grim Dawn", "Torchlight II", "Path of Exile",
            "Victor Vran", "Wolcen: Lords of Mayhem", "Warhammer: Chaosbane"
        ],
        "FPS": [
            "Half-Life 2", "DOOM Eternal", "Titanfall 2", "Halo Infinite",
            "Metro Exodus", "Far Cry 5", "Call of Duty: Modern Warfare"
        ],
        "RTS": [
            "Age of Empires II", "StarCraft II", "Total War: Warhammer II",
            "Company of Heroes", "Warcraft III", "Northgard", "Stronghold"
        ],
        "Simulation": [
            "Cities: Skylines", "The Sims 4", "Euro Truck Simulator 2",
            "Planet Coaster", "Football Manager 2023", "Farming Simulator 22"
        ],
        "Immersive Sim": [
        "System Shock", "Deus Ex", "Prey", "Dishonored"
        ],
        "Platformer": [
            "Celeste", "Cuphead", "Hollow Knight", "Ori and the Blind Forest",
            "Super Meat Boy", "Rayman Legends", "Little Nightmares"
        ],
        "Survival": [
            "Subnautica", "The Forest", "Rust", "Don't Starve",
            "Green Hell", "The Long Dark", "Valheim"
        ],
        "Puzzle": [
            "Portal 2", "The Witness", "Baba Is You", "Tetris Effect",
            "Talos Principle", "Limbo", "Inside"
        ],
        "Stealth": [
            "Dishonored", "Hitman 3", "Thief", "Splinter Cell: Blacklist",
            "Mark of the Ninja", "Aragami", "Sniper Elite 4"
        ],
        "Horror": [
            "Outlast", "Resident Evil 7", "Amnesia: Rebirth", "SOMA",
            "Dead Space", "Phasmophobia", "Visage"
        ],
        "Sandbox/Open World": [
            "Minecraft", "Skyrim", "Red Dead Redemption 2", "Just Cause 3",
            "Terraria", "No Man's Sky", "Mount & Blade: Warband"
        ],
        "Tactical": [
            "XCOM 2", "Into the Breach", "Battle Brothers", "Fire Emblem",
            "Gears Tactics", "Jagged Alliance 2", "Darkest Dungeon"
        ],
        "Metroidvania": [
            "Dead Cells", "Axiom Verge", "Blasphemous", "Hollow Knight",
            "Salt and Sanctuary", "Ender Lilies", "Bloodstained: Ritual of the Night"
        ],
        "Narrative Adventure": [
            "Alan Wake", "Life Is Strange", "Firewatch", "Oxenfree",
            "What Remains of Edith Finch", "The Walking Dead", "Gone Home"
        ],
        "Action Adventure": [
            "Tomb Raider", 
            "Uncharted 4", "God of War", "Sekiro: Shadows Die Twice",
            "Marvel's Spider-Man"
        ],
        "Visual Novel": [
            "Doki Doki Literature Club", "Clannad", "Steins;Gate",
            "Phoenix Wright: Ace Attorney", "Hatoful Boyfriend"
        ],
        "Rhythm": [
            "Beat Saber", "Thumper", "Crypt of the NecroDancer", "Dance Dance Revolution"
        ],
        "MMORPG": [
            "World of Warcraft", "Final Fantasy XIV", "The Elder Scrolls Online",
            "Guild Wars 2", "Black Desert Online"
        ],
        "Fighting": [
            "Street Fighter V", "Tekken 7", "Mortal Kombat 11", 
        ],
        "Roguelike/Roguelite": [
            "Hades", "Dead Cells", "Slay the Spire", "Enter the Gungeon", "The Binding of Isaac"
        ],
        "Cozy Life Sim": [
            "Stardew Valley",
        ],
        "Hacking / Terminal Puzzle": [
            "Hacknet", "TIS-100", "Exapunks", "Quadrilateral Cowboy"
        ],
        "Deckbuilder": [
            "Slay the Spire", "Monster Train", "Inscryption", "Griftlands"
        ],
        "Narrative Sci-Fi RPG": [
            "Citizen Sleeper", "Norco", "Neo Cab", "Tacoma"
        ],
        "Experimental": [
            "Critters for Sale", "Hylics", "Yume Nikki"
        ],
    }

    def genre_profile(tag_dicts):
        c = Counter()
        for d in tag_dicts:
            c.update(d)
        total = sum(c.values())
        return {k: v / total for k, v in c.items()} if total else {}

    genre_profiles = {g: genre_profile(df[df["name"].isin(games)]["tag_weights"]) for g, games in genre_examples.items()}

    # --- Assign Outliers Based on Nearest Cluster Within Matching Genre ---
    unassigned_mask = df["new_cluster"] == -1
    unassigned_indices = df.index[unassigned_mask]

    assigned_outliers = []
    for idx in unassigned_indices:
        tag_vec = df.loc[idx, "tag_weights"]

        # Find best genre match
        scores = {genre: weighted_jaccard(tag_vec, profile) for genre, profile in genre_profiles.items()}
        best_genre, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < 0.2:
            continue  

        # Get all cluster IDs that contain any exemplar from the genre
        valid_clusters = df[
            df["name"].isin(genre_examples[best_genre]) & (df["new_cluster"] != -1)
        ]["new_cluster"].unique()

        # Find all core games in those clusters
        candidate_indices = df[
            df["new_cluster"].isin(valid_clusters) & (df["new_cluster"] != -1)
        ].index

        if len(candidate_indices) == 0:
            continue

        candidate_embeddings = embedding[candidate_indices]
        outlier_embedding = embedding[idx].reshape(1, -1)

        closest_idx, _ = pairwise_distances_argmin_min(outlier_embedding, candidate_embeddings)
        closest_global_idx = candidate_indices[closest_idx[0]]
        assigned_cluster = df.loc[closest_global_idx, "new_cluster"]

        df.at[idx, "new_cluster"] = assigned_cluster
        assigned_outliers.append((df.at[idx, "name"], assigned_cluster, best_genre, round(best_score, 3)))

    print("Genre-guided outlier reassignments:")
    for name, cluster, genre, score in assigned_outliers:
        print(f"{name} → Cluster {cluster} ({genre}, score={score})")

    # --- Cluster Quality Analysis ---
    def tag_entropy(cluster_tags):
        all_tags = [tag for tags in cluster_tags for tag in tags]
        tag_counts = Counter(all_tags)
        total = sum(tag_counts.values())
        probs = np.array([count / total for count in tag_counts.values()])
        entropy = -np.sum(probs * np.log2(probs)) if total > 0 else 0
        return entropy

    def tag_purity(cluster_tags):
        all_tags = [tag for tags in cluster_tags for tag in tags]
        if not all_tags:
            return 0
        most_common = Counter(all_tags).most_common(1)[0][1]
        return most_common / len(all_tags)

    def evaluate_cluster_quality(df, cluster_col="new_cluster", tag_col="tags", top_n=10):
        quality = []
        for cluster_id, group in df.groupby(cluster_col):
            tags = group[tag_col].tolist()
            entropy = tag_entropy(tags)
            purity = tag_purity(tags)
            size = len(tags)
            quality.append((cluster_id, entropy, purity, size))

        quality.sort(key=lambda x: (-x[1], x[2]))
        return quality

    def reassign_cluster_by_genre(df, embedding, cluster_id, genre_profiles, genre_examples, full_df=None, cluster_col="new_cluster", tag_col="tag_weights", min_score=0.15):
        subset_indices = df[df[cluster_col] == cluster_id].index
        assigned_outliers = []

        for idx in subset_indices:
            tag_vec = df.loc[idx, tag_col]

            # Find best genre match
            scores = {genre: weighted_jaccard(tag_vec, profile) for genre, profile in genre_profiles.items()}

            for genre, score in sorted(scores.items(), key=lambda x: -x[1])[:3]:
                best_genre, best_score = max(scores.items(), key=lambda x: x[1])

            if best_score < min_score:
                continue  

            # Valid clusters with exemplar games in best genre
            source_df = df if full_df is None else full_df
            valid_clusters = source_df[
                source_df["name"].isin(genre_examples[best_genre]) & (source_df[cluster_col] != -1)
            ][cluster_col].unique()

            # Candidate indices in those clusters
            candidate_indices = source_df[
                source_df[cluster_col].isin(valid_clusters) & (source_df[cluster_col] != -1)
            ].index

            if len(candidate_indices) == 0:
                continue

            candidate_embeddings = embedding[candidate_indices]
            game_embedding = embedding[idx].reshape(1, -1)

            closest_idx, _ = pairwise_distances_argmin_min(game_embedding, candidate_embeddings)
            closest_global_idx = candidate_indices[closest_idx[0]]
            assigned_cluster = df.loc[closest_global_idx, cluster_col]

            # Assign new cluster
            df.at[idx, cluster_col] = assigned_cluster
            assigned_outliers.append((df.at[idx, "name"], assigned_cluster, best_genre, round(best_score, 3)))

        print(f"Reassigned {len(assigned_outliers)} games from cluster {cluster_id} by genre similarity.")
        return assigned_outliers

    def loop_subcluster_by_quality(df, embedding, cluster_col="new_cluster", tag_col="tags",
                                   top_n=5, eps=0.15, min_samples=3, size_threshold=10, max_iters=10):
        for iteration in range(max_iters):
            print(f"\nIteration {iteration+1}")

            # Evaluate cluster quality
            weak_clusters = evaluate_cluster_quality(df, cluster_col=cluster_col, tag_col=tag_col)

            # Filter top_n worst clusters that are larger than size_threshold
            clusters_to_subcluster = [(cid, entropy, purity, size) for cid, entropy, purity, size in weak_clusters[:top_n] if size > size_threshold]

            if not clusters_to_subcluster:
                print("No clusters above size threshold to subcluster. Stopping.")
                break

            print(f"Clusters to subcluster this iteration: {[cid for cid, _, _, _ in clusters_to_subcluster]}")

            max_label = df[cluster_col].max() + 1
            updated = False

            for cid, _, _, _ in clusters_to_subcluster:
                subset_idx = df[df[cluster_col] == cid].index
                subset_emb = embedding[subset_idx]

                sub_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(subset_emb)

                unique_sub_labels = set(sub_labels)
                unique_sub_labels.discard(-1)  # ignore noise

                if len(unique_sub_labels) <= 1:
                    print(f"Cluster {cid} could not be subclustered further. trying genres")
                    assigned = reassign_cluster_by_genre(df, embedding, cid, genre_profiles, genre_examples, full_df=df)
                    if assigned:
                        updated = True
                    continue

                # Assign new unique cluster labels
                label_map = {lbl: max_label + i for i, lbl in enumerate(unique_sub_labels)}
                max_label += len(unique_sub_labels)

                for i, lbl in zip(subset_idx, sub_labels):
                    if lbl != -1:
                        df.at[i, cluster_col] = label_map[lbl]

                updated = True

            if not updated:
                print("No clusters were updated this iteration. Stopping.")
                break

        print("Subclustering complete.")
        return df

    df = loop_subcluster_by_quality(df, embedding,cluster_col="new_cluster", tag_col="tags",top_n=5,eps=0.15,min_samples=3,size_threshold=10,max_iters=10)

    empty_tags_mask = df["tags"].apply(lambda x: len(x) == 0)
    df.loc[empty_tags_mask, "new_cluster"] = -1

    df.to_csv("steam_all_processed.csv", index=False)

