# Steam-Backlog-Cleaner

A personalized Steam game recommendation app that analyzes your Steam library based on playtime, tags, and developers to suggest games you may enjoy.

---

## Features

- Fetch your Steam library using your **SteamID64** and **Steam API key**.
- Analyze game data: playtime, achievements, tags, genres, developers.
- Recommend games using **hybrid scoring**:
  - Embedding similarity  
  - Tag overlap  
  - Developer affinity  
  - Cluster-based bonuses
- Generate **explainable recommendations** for transparency.
- Interactive UI built with **Streamlit**.

---

## Getting Started

### Prerequisites

- Python 3.8+
- A valid Steam API key  
  → [Get a Steam API key](https://steamcommunity.com/dev/apikey)  
- Your SteamID64  
  → [Find your SteamID64](https://steamid.io/)

---

### Installation

1. **Clone this repository:**
   
   ```bash
   git clone https://github.com/emirshn/Steam-Backlog-Cleaner.git
   cd steam-backlog-cleaner
   
2. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
    
3. **Set up environment variables:**

    ```bash
    cp .env.example .env

4. Open .env and replace placeholder
   ```env
   STEAM_API_KEY=your_real_api_key_here
  
### Usage
* Run the app using Streamlit:
  ```bash
  streamlit run app.py

Then follow the steps in your browser:

- Enter your SteamID64.

- Fetch your Steam library.

- Choose a recommendation mode:

  - Based on a specific game name

  - Based on your liked games

- Adjust filters:

  - Minimum playtime
  
  - Minimum achievement percentage

- View personalized recommendations with reasons and insights.

## Example Usage
<img width="1516" height="735" alt="image" src="https://github.com/user-attachments/assets/41c9bdf4-4bc4-45ac-b574-de1925f30497" />
<img width="1484" height="449" alt="image" src="https://github.com/user-attachments/assets/641131e4-2043-499b-8caf-9ad0c9c3f3c8" />

## Known Issues
* Recommendation accuracy is not perfect. While the system uses embeddings, developer info, and tag similarities, some suggestions may still seem off or irrelevant.
* Heuristic thresholds (e.g., for playtime or achievements) may need user tuning for better results.
* Tag or genre data can sometimes be noisy or inconsistent depending on the game's Steam metadata because of its based on steam user data.
* Some games doesnt have tags in their metadata, these games are mostly f2p games and they are counted as noises.
* Fetching large libraries may take a few minutes.
