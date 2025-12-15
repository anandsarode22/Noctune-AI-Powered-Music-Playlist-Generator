# app_recommender.py
"""
Flask recommender server (ready-to-run).

Notes:
- Uses trained artifacts from ./artifacts (supports joblib or pickle naming).
- Removes 'album' and 'duration' from all API responses and CSV download.
- Accepts uploads as 'file' (single) or 'files[]' (multiple).
- Requires librosa only for audio uploads (optional for search-by-title/genre).
- Enable CORS for cross-origin frontend calls.
"""
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pathlib import Path
from typing import List, Dict
from io import StringIO, BytesIO
import os
import re
import json
import csv
import sys

import numpy as np
import pandas as pd

# Prefer joblib for sklearn artifacts, but support pickle as fallback
import joblib
import pickle

# Optional audio deps
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available. Audio feature extraction disabled.")

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths: try to be compatible with both tester and your prior flask naming
BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
# tester artifact names:
SCALER_JOBLIB = ART / "scaler.joblib"
NN_JOBLIB = ART / "nn.joblib"       # maybe a dict {"nn":..., "pca":...}
FEAT_JSON = ART / "feature_cols.json"
META_SAMPLE = ART.parent / "metadata_sample.csv"
FEATURES_CSV = BASE / "fma_features_processed.csv"
FALLBACK_META = BASE / "fma_song_metadata.csv"

# older/pickle-named options (your previous app used these):
NN_PKL = ART / "fma_nn.pkl"
SCALER_PKL = ART / "fma_scaler.pkl"
FEATURES_PKL = ART / "fma_feature_cols.pkl"

# Cached artifacts
ARTIFACTS = None

TITLE_COLS = ["title", "track_title", "track"]
ARTIST_COLS = ["artist", "artist_name", "artists"]
GENRE_COLS = ["genre", "genre_top", "primary_genre"]
ID_COLS = ["track_id", "id"]

def _normalize_name(s: str):
    if s is None:
        return ""
    s2 = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).lower()).strip("_")
    return re.sub(r"_+", "_", s2)

def _ensure_vec_compatible_with_scaler(vec: np.ndarray, scaler) -> np.ndarray:
    expected = getattr(scaler, "n_features_in_", None)
    if expected is None:
        return vec
    n = vec.shape[1]
    if n == expected:
        return vec
    if n < expected:
        pad = np.zeros((1, expected - n), dtype=vec.dtype)
        print(f"⚠️ Padding {expected - n} zeros to match scaler ({n}→{expected})")
        return np.hstack([vec, pad])
    print(f"⚠️ Trimming {n - expected} extras to match scaler ({n}→{expected})")
    return vec[:, :expected]

def smart_map_features_to_vector(features_dict: Dict[str, float], feature_cols: List[str]):
    norm_to_key = {_normalize_name(k): k for k in features_dict.keys()}

    def find_best_match(col: str):
        norm_col = _normalize_name(col)
        if norm_col in norm_to_key:
            return norm_to_key[norm_col]
        # fuzzy family matches
        for fam in ["mfcc", "chroma", "spec", "rolloff", "zcr", "rms", "tempo", "mel"]:
            if fam in norm_col:
                for nk in norm_to_key:
                    if fam in nk:
                        return norm_to_key[nk]
        return None

    vec = []
    for col in feature_cols:
        if any(k in col.lower() for k in ["genre", "label"]):
            vec.append(0.0)
            continue
        match = find_best_match(col)
        if match is None:
            vec.append(0.0)
            continue
        val = features_dict.get(match, 0.0)
        try:
            arr = np.asarray(val)
            if arr.ndim > 0:
                val_f = float(arr.mean())
            else:
                val_f = float(arr)
        except Exception:
            try:
                val_f = float(val)
            except Exception:
                val_f = 0.0
        vec.append(val_f)
    return np.array(vec, dtype=float).reshape(1, -1)

def extract_features_from_audio(audio_path: str, sr=22050):
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa not installed. Run: pip install librosa soundfile")
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    y, sr = librosa.load(str(p), sr=sr, mono=True)
    feats = {}
    feats["tempo"], _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)))
    feats["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, v in enumerate(np.mean(mfcc, axis=1), 1):
        feats[f"mfcc{i}_mean"] = float(v)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma, axis=1), 1):
        feats[f"chroma{i}_mean"] = float(v)
    feats["mel_mean"] = float(np.mean(librosa.feature.melspectrogram(y=y, sr=sr)))
    return feats

# ---- Artifact loading (robust) ----
def load_artifacts():
    """
    Load model artifacts. Supports both:
      - joblib layout: artifacts/scaler.joblib, artifacts/nn.joblib, artifacts/feature_cols.json
      - pickle layout: artifacts/fma_scaler.pkl, artifacts/fma_nn.pkl, artifacts/fma_feature_cols.pkl
    Also loads FEATURES_CSV and metadata CSVs (metadata_sample.csv / fallback).
    """
    global ARTIFACTS
    if ARTIFACTS is not None:
        return ARTIFACTS

    scaler = None
    nn = None
    pca = None
    feature_cols = None

    try:
        if SCALER_JOBLIB.exists() and NN_JOBLIB.exists() and FEAT_JSON.exists():
            scaler = joblib.load(SCALER_JOBLIB)
            nn_bundle = joblib.load(NN_JOBLIB)
            if isinstance(nn_bundle, dict):
                nn = nn_bundle.get("nn") or nn_bundle.get("model") or nn_bundle.get("nearest_neighbors")
                pca = nn_bundle.get("pca")
            else:
                nn = nn_bundle
            feature_cols = json.loads(FEAT_JSON.read_text(encoding="utf8"))
            print("Loaded artifacts (joblib/json) from artifacts/")
        elif NN_PKL.exists() and SCALER_PKL.exists() and FEATURES_PKL.exists():
            with open(NN_PKL, "rb") as f:
                nn_bundle = pickle.load(f)
            with open(SCALER_PKL, "rb") as f:
                scaler = pickle.load(f)
            with open(FEATURES_PKL, "rb") as f:
                feature_cols = pickle.load(f)
            if isinstance(nn_bundle, dict):
                nn = nn_bundle.get("model") or nn_bundle.get("nn") or nn_bundle.get("nearest_neighbors")
                pca = nn_bundle.get("pca")
            else:
                nn = nn_bundle
            print("Loaded artifacts (pickle) from artifacts/")
        else:
            # Try scanning the artifacts folder for plausible files
            for p in ART.iterdir() if ART.exists() else []:
                n = p.name.lower()
                if ("scaler" in n or "std" in n) and p.suffix in (".joblib", ".pkl"):
                    try:
                        scaler = joblib.load(p) if p.suffix == ".joblib" else pickle.load(open(p, "rb"))
                    except Exception:
                        pass
                if ("nn" in n or "nearest" in n) and p.suffix in (".joblib", ".pkl"):
                    try:
                        nn_bundle = joblib.load(p) if p.suffix == ".joblib" else pickle.load(open(p, "rb"))
                        if isinstance(nn_bundle, dict):
                            nn = nn_bundle.get("nn") or nn_bundle.get("model") or nn_bundle.get("nearest_neighbors")
                            pca = nn_bundle.get("pca")
                        else:
                            nn = nn_bundle
                    except Exception:
                        pass
                if "feature" in n and p.suffix in (".json", ".pkl", ".joblib"):
                    try:
                        if p.suffix == ".json":
                            feature_cols = json.loads(p.read_text(encoding="utf8"))
                        else:
                            feature_cols = joblib.load(p) if p.suffix == ".joblib" else pickle.load(open(p, "rb"))
                    except Exception:
                        pass

    except Exception as e:
        print("Error loading artifacts:", e)
        scaler = None
        nn = None
        feature_cols = None

    # Load dataframes
    df = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else None
    meta_sample = pd.read_csv(META_SAMPLE) if META_SAMPLE.exists() else None
    meta_fallback = pd.read_csv(FALLBACK_META) if FALLBACK_META.exists() else None

    # Normalize/merge metadata similar to tester
    def pick_first_col(cols, frame):
        for c in cols:
            if frame is not None and c in frame.columns:
                return c
        return None

    def normalize_meta(frame):
        if frame is None:
            return None
        col_title = pick_first_col(TITLE_COLS, frame)
        col_artist = pick_first_col(ARTIST_COLS, frame)
        col_genre = pick_first_col(GENRE_COLS, frame)
        col_id = pick_first_col(ID_COLS, frame)
        keep = [c for c in [col_id, col_title, col_artist, col_genre] if c]
        if not keep:
            return pd.DataFrame(index=frame.index)
        out = frame[keep].copy()
        rename_map = {}
        if col_id and col_id != "track_id": rename_map[col_id] = "track_id"
        if col_title and col_title != "title": rename_map[col_title] = "title"
        if col_artist and col_artist != "artist": rename_map[col_artist] = "artist"
        if col_genre and col_genre != "genre": rename_map[col_genre] = "genre"
        if rename_map:
            out.rename(columns=rename_map, inplace=True)
        return out

    ms = normalize_meta(meta_sample) if meta_sample is not None else None
    mf = normalize_meta(meta_fallback) if meta_fallback is not None else None

    meta = None
    # Merge strategies as in tester
    if ms is not None and mf is not None and "track_id" in ms.columns and "track_id" in mf.columns:
        meta = ms.merge(mf, on="track_id", how="left", suffixes=("", "_fb"))
        for c in ["title", "artist", "genre"]:
            if c in meta.columns and f"{c}_fb" in meta.columns:
                meta[c] = meta[c].fillna(meta[f"{c}_fb"])
        meta = meta[[c for c in meta.columns if not c.endswith("_fb")]]
    elif ms is not None and ("title" not in ms.columns or "artist" not in ms.columns) and mf is not None and len(mf) == (len(df) if df is not None else len(mf)):
        meta = mf
    else:
        cand = []
        for fr in [ms, mf]:
            if fr is None:
                continue
            score = sum(c in fr.columns for c in ["title", "artist"]) * 2 + sum(c in fr.columns for c in ["genre", "track_id"])
            nnz = 0
            for c in ["title", "artist"]:
                if c in fr.columns:
                    nnz += fr[c].notna().sum()
            cand.append((score, nnz, fr))
        if cand:
            cand.sort(reverse=True)
            meta = cand[0][2]

    # Remove label-like columns from feature order if any slipped in
    if feature_cols:
        feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ["genre", "label"])]

    ARTIFACTS = {
        "scaler": scaler,
        "nn": nn,
        "pca": pca,
        "feature_cols": feature_cols,
        "df": df,
        "meta": meta
    }
    return ARTIFACTS

# ---- Transform helper ----
def transform_vec(vec_raw: np.ndarray, artifacts: Dict) -> np.ndarray:
    scaler = artifacts.get("scaler")
    pca = artifacts.get("pca")
    vec = _ensure_vec_compatible_with_scaler(np.array(vec_raw).reshape(1, -1).astype(np.float32), scaler) if scaler is not None else np.array(vec_raw).reshape(1, -1).astype(np.float32)
    if scaler is not None:
        Xs = scaler.transform(vec)
    else:
        Xs = vec
    if pca is not None:
        try:
            Xs = pca.transform(Xs)
        except Exception:
            pass
    return Xs

# ---- Recommendation core ----
def recommend_from_vector(vec_raw, artifacts, top_k=10):
    nn = artifacts.get("nn")
    if nn is None:
        raise RuntimeError("NearestNeighbors model not loaded.")
    meta = artifacts.get("meta")
    df = artifacts.get("df")
    fcols = artifacts.get("feature_cols")

    Xs = transform_vec(vec_raw, artifacts)
    dists, inds = nn.kneighbors(Xs, n_neighbors=top_k + 1)
    inds, dists = inds[0][1:], dists[0][1:]

    results = []
    for rank, (i, d) in enumerate(zip(inds, dists), start=1):
        info = {"rank": rank, "index": int(i), "distance": float(d)}
        if isinstance(meta, pd.DataFrame) and 0 <= int(i) < len(meta):
            row = meta.iloc[int(i)].to_dict()
            info["meta"] = {k: row.get(k) for k in ["track_id", "title", "artist", "genre"] if k in row}
        else:
            if isinstance(df, pd.DataFrame) and fcols and 0 <= int(i) < len(df):
                info["meta"] = {"index": int(i)}
            else:
                info["meta"] = {}
        results.append(info)
    return results

# ---- Convenience recommenders (tester-like) ----
def search_by_uploaded_song(filepath: str, artifacts: Dict, top_k: int = 10):
    feats = extract_features_from_audio(filepath)
    mapped = smart_map_features_to_vector(feats, artifacts["feature_cols"])
    return recommend_from_vector(mapped.flatten(), artifacts, top_k=top_k)

def search_by_typed_song(title: str, artifacts: Dict, top_k: int = 10):
    meta = artifacts.get("meta")
    df = artifacts.get("df")
    fcols = artifacts.get("feature_cols")
    if not isinstance(meta, pd.DataFrame) or "title" not in meta.columns:
        raise RuntimeError("No metadata with titles available.")
    titles = meta["title"].astype(str)
    mask = titles.str.lower() == title.lower()
    if not mask.any():
        # close matches
        suggestions = []
        try:
            import difflib
            suggestions = difflib.get_close_matches(title, titles.tolist(), n=5, cutoff=0.6)
        except Exception:
            suggestions = []
        raise ValueError("Song title not found. Suggestions: " + ", ".join(suggestions) if suggestions else "Song title not found.")
    idx = mask[mask].index[0]
    if df is None:
        raise RuntimeError("Feature dataframe not available.")
    vec = df[fcols].iloc[idx].fillna(0).values
    return recommend_from_vector(vec, artifacts, top_k=top_k)

def discover_by_genre(genre: str, artifacts: Dict, top_k: int = 10):
    meta = artifacts.get("meta")
    df = artifacts.get("df")
    fcols = artifacts.get("feature_cols")
    if not isinstance(meta, pd.DataFrame) or "genre" not in meta.columns:
        raise RuntimeError("No metadata with a genre/genre_top column available.")
    mask = meta["genre"].astype(str).str.lower() == genre.lower()
    idxs = meta[mask].index.tolist()
    if not idxs:
        import difflib
        genres = sorted(set(meta["genre"].dropna().astype(str)))
        suggestions = difflib.get_close_matches(genre, genres, n=5, cutoff=0.6)
        raise ValueError("Genre not found. Suggestions: " + ", ".join(suggestions) if suggestions else "Genre not found.")
    centroid = np.mean(df[fcols].iloc[idxs].fillna(0).values, axis=0)
    return recommend_from_vector(centroid, artifacts, top_k=top_k)

def discover_by_artist(artist: str, artifacts: Dict, top_k: int = 10):
    meta = artifacts.get("meta")
    df = artifacts.get("df")
    fcols = artifacts.get("feature_cols")
    if not isinstance(meta, pd.DataFrame) or "artist" not in meta.columns:
        raise RuntimeError("No metadata with an artist column available.")
    mask = meta["artist"].astype(str).str.lower() == artist.lower()
    idxs = meta[mask].index.tolist()
    if not idxs:
        import difflib
        artists = sorted(set(meta["artist"].dropna().astype(str)))
        suggestions = difflib.get_close_matches(artist, artists, n=5, cutoff=0.6)
        raise ValueError("Artist not found. Suggestions: " + ", ".join(suggestions) if suggestions else "Artist not found.")
    centroid = np.mean(df[fcols].iloc[idxs].fillna(0).values, axis=0)
    return recommend_from_vector(centroid, artifacts, top_k=top_k)

# ---- Utility links ----
def generate_youtube_link(title, artist):
    if not title or not artist:
        return ""
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"

def generate_spotify_link(title, artist):
    if not title or not artist:
        return ""
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://open.spotify.com/search/{query}"

# ---- Flask endpoints ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # If you have a template 'index.html' in templates/, it will be served.
    # Otherwise a basic JSON will indicate the service is up.
    index_path = BASE / "templates" / "index.html"
    if index_path.exists():
        return render_template('index.html')
    return jsonify({'status': 'Music Playlist Generator running'})

@app.route('/generate-playlist', methods=['POST'])
def generate_playlist():
    """
    Improved upload handler:
      - accepts multipart uploads under 'files[]' (multiple) or 'file' (single)
      - logs debug info and returns detailed error messages
      - returns recommendations WITHOUT 'album' and 'duration'
    """
    try:
        artifacts = load_artifacts()
        if artifacts is None or artifacts.get("nn") is None:
            return jsonify({'error': 'Recommender system not available. Please ensure model artifacts are present.'}), 500

        # Accept either 'files[]' (from JS form with multiple) or 'file' (single file)
        files = []
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
        elif 'file' in request.files:
            files = [request.files.get('file')]
        else:
            # Some frontends send other names — try to find any file part
            for k in request.files:
                files = request.files.getlist(k)
                if files:
                    break

        if not files or all(f.filename == '' for f in files):
            app.logger.debug("No files present in request.files keys: %s", list(request.files.keys()))
            return jsonify({'error': 'No files uploaded. Make sure the form uses enctype="multipart/form-data" and field name is files[] or file.'}), 400

        app.logger.debug("Received %d file(s) for processing", len(files))

        saved_files = []
        all_recommendations = []

        for file in files:
            filename = file.filename or ""
            if filename == "":
                app.logger.warning("Skipping empty filename part")
                continue

            if not allowed_file(filename):
                app.logger.warning("Rejected disallowed extension for file: %s", filename)
                continue

            safe_name = secure_filename(filename)
            dest = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

            try:
                file.save(dest)
                saved_files.append(dest)
                app.logger.info("Saved uploaded file to %s", dest)
            except Exception as e:
                app.logger.exception("Failed to save uploaded file %s: %s", filename, e)
                continue

            # Try to extract features and recommend
            try:
                features = extract_features_from_audio(dest)
                mapped_vec = smart_map_features_to_vector(features, artifacts["feature_cols"])
                recs = recommend_from_vector(mapped_vec.flatten(), artifacts, top_k=10)
                # convert to a friendly shape (no album, no duration)
                for r in recs:
                    m = r.get("meta", {}) or {}
                    title = m.get("title", "") or ""
                    artist = m.get("artist", "") or ""
                    all_recommendations.append({
                        "name": title,
                        "artist": artist,
                        "genre": m.get("genre", ""),
                        "distance": r.get("distance"),
                        "index": r.get("index"),
                        "youtube_link": generate_youtube_link(title, artist),
                        "spotify_link": generate_spotify_link(title, artist)
                    })
            except Exception as e:
                app.logger.exception("Failed to process file %s: %s", dest, e)
                continue

        # cleanup saved files
        for p in saved_files:
            try:
                os.remove(p)
            except Exception:
                app.logger.debug("Failed to remove temp file %s", p)

        if not all_recommendations:
            return jsonify({'error': 'Could not generate recommendations from uploaded files'}), 400

        # dedupe by name+artist
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            key = (rec.get('name',''), rec.get('artist',''))
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)

        return jsonify({
            'success': True,
            'playlist': unique_recs[:20],
            'message': f'Processed {len(saved_files)} audio files, found {len(unique_recs)} unique recommendations'
        })
    except Exception as e:
        app.logger.exception("Unexpected error in generate_playlist: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/search-song', methods=['POST'])
def search_song():
    """
    Search for a song and get recommendations
    """
    try:
        data = request.get_json(force=True)
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'No search query provided'}), 400

        artifacts = load_artifacts()
        if artifacts is None or artifacts.get("nn") is None:
            return jsonify({'error': 'Recommender system not available'}), 500

        meta = artifacts["meta"]
        if meta is None:
            return jsonify({'error': 'Metadata not available'}), 500

        # Search for song in metadata
        query_lower = query.lower()
        matches = meta[
            meta["title"].astype(str).str.lower().str.contains(query_lower, na=False) |
            meta["artist"].astype(str).str.lower().str.contains(query_lower, na=False)
        ]

        if len(matches) == 0:
            return jsonify({'error': f'No songs found matching \"{query}\"'}), 404

        # Get first match
        idx = matches.index[0]
        song_info = meta.iloc[idx].to_dict()

        # Get feature vector for this song
        df = artifacts.get("df")
        fcols = artifacts.get("feature_cols")
        if df is None or fcols is None:
            return jsonify({'error': 'Feature dataframe not available'}), 500

        vec = df[fcols].iloc[idx].fillna(0).values

        # Get recommendations
        recommendations = recommend_from_vector(vec, artifacts, top_k=15)

        # friendly formatting (no album, no duration)
        playlist = []
        for r in recommendations:
            m = r.get("meta", {}) or {}
            playlist.append({
                "name": m.get("title", ""),
                "artist": m.get("artist", ""),
                "genre": m.get("genre", ""),
                "distance": r.get("distance"),
                "index": r.get("index"),
                "youtube_link": generate_youtube_link(m.get("title", ""), m.get("artist", "")),
                "spotify_link": generate_spotify_link(m.get("title", ""), m.get("artist", ""))
            })

        return jsonify({
            'success': True,
            'query_song': {
                'name': song_info.get('title', 'Unknown'),
                'artist': song_info.get('artist', 'Unknown Artist'),
                'genre': song_info.get('genre', 'Unknown')
            },
            'playlist': playlist,
            'message': f'Found recommendations based on \"{song_info.get("title", "Unknown")}\" by {song_info.get("artist", "Unknown")}'
        })

    except Exception as e:
        app.logger.exception("Error in search_song: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/get-genres', methods=['GET'])
def get_genres():
    """
    Get list of available genres
    """
    try:
        artifacts = load_artifacts()
        if artifacts is None or artifacts.get("meta") is None:
            return jsonify({'genres': []})
        genres = sorted(artifacts["meta"]["genre"].dropna().unique().tolist())
        return jsonify({'genres': genres})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend-by-genre', methods=['POST'])
def recommend_by_genre():
    """
    Get recommendations based on genre centroid
    """
    try:
        data = request.get_json(force=True)
        genre = data.get('genre', '').strip()

        if not genre:
            return jsonify({'error': 'No genre provided'}), 400

        artifacts = load_artifacts()
        if artifacts is None or artifacts.get("nn") is None:
            return jsonify({'error': 'Recommender system not available'}), 500

        meta = artifacts.get("meta")
        df = artifacts.get("df")
        fcols = artifacts.get("feature_cols")

        if meta is None or df is None or fcols is None:
            return jsonify({'error': 'Required data not available'}), 500

        idxs = meta[meta["genre"].astype(str).str.lower() == genre.lower()].index.tolist()

        if not idxs:
            return jsonify({'error': f'Genre \"{genre}\" not found'}), 404

        centroid = np.mean(df[fcols].iloc[idxs].fillna(0).values, axis=0)
        recs = recommend_from_vector(centroid, artifacts, top_k=15)

        playlist = []
        for r in recs:
            m = r.get("meta", {}) or {}
            playlist.append({
                "name": m.get("title", ""),
                "artist": m.get("artist", ""),
                "genre": m.get("genre", ""),
                "distance": r.get("distance"),
                "index": r.get("index"),
                "youtube_link": generate_youtube_link(m.get("title", ""), m.get("artist", "")),
                "spotify_link": generate_spotify_link(m.get("title", ""), m.get("artist", ""))
            })

        return jsonify({
            'success': True,
            'genre': genre,
            'playlist': playlist,
            'message': f'Generated playlist for {genre} genre'
        })
    except Exception as e:
        app.logger.exception("Error in recommend_by_genre: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/download-csv', methods=['POST'])
def download_csv():
    """
    Convert playlist JSON to CSV and send as download (no album/duration columns)
    """
    try:
        data = request.get_json(force=True)
        playlist = data.get('playlist', [])

        if not playlist:
            return jsonify({'error': 'No playlist data provided'}), 400

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header (no album, no duration)
        writer.writerow(['Track #', 'Name', 'Artist', 'Genre', 'YouTube Link', 'Spotify Link'])

        # Write data
        for idx, track in enumerate(playlist, 1):
            writer.writerow([
                idx,
                track.get('name', ''),
                track.get('artist', ''),
                track.get('genre', 'N/A'),
                track.get('youtube_link', ''),
                track.get('spotify_link', '')
            ])

        # Convert to bytes
        output.seek(0)
        bytes_output = BytesIO()
        bytes_output.write(output.getvalue().encode('utf-8'))
        bytes_output.seek(0)

        return send_file(
            bytes_output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='playlist.csv'
        )

    except Exception as e:
        app.logger.exception("Error in download_csv: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/download-json', methods=['POST'])
def download_json():
    """
    Send playlist as JSON file download (unchanged keys; no album/duration expected)
    """
    try:
        data = request.get_json(force=True)
        playlist = data.get('playlist', [])

        if not playlist:
            return jsonify({'error': 'No playlist data provided'}), 400

        # Create JSON in memory
        json_data = json.dumps({'playlist': playlist}, indent=2)
        bytes_output = BytesIO(json_data.encode('utf-8'))
        bytes_output.seek(0)

        return send_file(
            bytes_output,
            mimetype='application/json',
            as_attachment=True,
            download_name='playlist.json'
        )
    except Exception as e:
        app.logger.exception("Error in download_json: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting Music Playlist Generator...")
    art = load_artifacts()
    if art and art.get("meta") is not None:
        try:
            print("Metadata columns:", art["meta"].columns.tolist())
        except Exception:
            pass
    app.run(debug=True, host='0.0.0.0', port=5000)
