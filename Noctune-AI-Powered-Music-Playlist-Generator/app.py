from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import json
import csv
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
import pickle
import re
from pathlib import Path
from typing import List, Dict

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths to recommender artifacts
BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
FEATURES_CSV = BASE / "fma_features_processed.csv"
METADATA_CSV = BASE / "fma_song_metadata.csv"
NN_PKL = ART / "fma_nn.pkl"
SCALER_PKL = ART / "fma_scaler.pkl"
FEATURES_PKL = ART / "fma_feature_cols.pkl"

ARTIFACTS = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available. Audio feature extraction disabled.")

def _normalize_name(s: str):
    """Normalize a feature name: lowercase, underscores only."""
    if s is None:
        return ""
    s2 = re.sub(r'[^0-9a-zA-Z]+', '_', str(s).lower()).strip('_')
    return re.sub(r'_+', '_', s2)

def _ensure_vec_compatible_with_scaler(vec: np.ndarray, scaler) -> np.ndarray:
    """Pad or trim feature vector to match scaler input size."""
    expected = getattr(scaler, "n_features_in_", None)
    if expected is None:
        return vec
    n = vec.shape[1]
    if n == expected:
        return vec
    if n < expected:
        diff = expected - n
        pad = np.zeros((1, diff), dtype=vec.dtype)
        print(f"⚠️ Padding {diff} zeros to match scaler ({n}→{expected})")
        return np.hstack([vec, pad])
    print(f"⚠️ Trimming {n - expected} extras to match scaler ({n}→{expected})")
    return vec[:, :expected]

def smart_map_features_to_vector(features_dict: Dict[str, float], feature_cols: List[str]):
    """Map extracted audio features to the correct model feature order."""
    norm_to_key = {_normalize_name(k): k for k in features_dict.keys()}

    def find_best_match(col: str):
        norm_col = _normalize_name(col)
        if norm_col in norm_to_key:
            return norm_to_key[norm_col]
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
        vec.append(float(features_dict.get(match, 0.0)) if match else 0.0)
    return np.array(vec, dtype=float).reshape(1, -1)

def extract_features_from_audio(audio_path: str, sr=22050):
    """Extract audio features using librosa."""
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa not installed. Run: pip install librosa soundfile")
    
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    feats = {}
    
    # Extract features
    feats["tempo"], _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))
    feats["rms_mean"] = np.mean(librosa.feature.rms(y=y))
    feats["spectral_rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, v in enumerate(np.mean(mfcc, axis=1), 1):
        feats[f"mfcc{i}_mean"] = v
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma, axis=1), 1):
        feats[f"chroma{i}_mean"] = v
    
    # Mel spectrogram
    feats["mel_mean"] = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
    
    return feats

# ---------------------------------------
# Load recommender artifacts
# ---------------------------------------
def load_artifacts():
    """Load trained recommender model artifacts."""
    global ARTIFACTS
    
    if ARTIFACTS is not None:
        return ARTIFACTS
    
    if not NN_PKL.exists() or not SCALER_PKL.exists() or not FEATURES_PKL.exists():
        print("⚠️ Missing model artifacts. Recommender system will not be available.")
        return None
    
    try:
        with open(NN_PKL, "rb") as f:
            nn_data = pickle.load(f)
        with open(SCALER_PKL, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PKL, "rb") as f:
            feature_cols = pickle.load(f)
        
        # Remove label columns
        feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ["genre", "label"])]
        
        # Load metadata and features
        meta = pd.read_csv(METADATA_CSV) if METADATA_CSV.exists() else None
        df = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else None
        
        ARTIFACTS = {
            "nn": nn_data["model"],
            "scaler": scaler,
            "feature_cols": feature_cols,
            "meta": meta,
            "features_df": df
        }
        
        print("✅ Recommender artifacts loaded successfully.")
        return ARTIFACTS
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return None

# ---------------------------------------
# Recommender functions
# ---------------------------------------
def recommend_from_vector(vec_raw, artifacts, top_k=10):
    """Generate recommendations from feature vector."""
    scaler = artifacts["scaler"]
    nn = artifacts["nn"]
    meta = artifacts["meta"]
    
    vec = _ensure_vec_compatible_with_scaler(np.array(vec_raw).reshape(1, -1), scaler)
    vec_scaled = scaler.transform(vec)
    
    # Get nearest neighbors
    dists, inds = nn.kneighbors(vec_scaled, n_neighbors=top_k + 1)
    inds, dists = inds[0][1:], dists[0][1:]
    
    results = []
    for i, d in zip(inds, dists):
        info = {}
        if meta is not None and i < len(meta):
            row = meta.iloc[i].to_dict()
            info = {
                "name": row.get("title", "Unknown"),
                "artist": row.get("artist", "Unknown Artist"),
                "album": row.get("album", "Unknown Album"),
                "genre": row.get("genre", "Unknown"),
                "duration": format_duration(row.get("duration", 0)),
                "youtube_link": generate_youtube_link(row.get("title"), row.get("artist")),
                "spotify_link": generate_spotify_link(row.get("title"), row.get("artist"))
            }
        results.append(info)
    
    return results

def format_duration(seconds):
    """Format duration from seconds to MM:SS."""
    try:
        seconds = int(float(seconds))
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"
    except:
        return "0:00"

def generate_youtube_link(title, artist):
    """Generate YouTube search link."""
    if not title or not artist:
        return ""
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"

def generate_spotify_link(title, artist):
    """Generate Spotify search link."""
    if not title or not artist:
        return ""
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://open.spotify.com/search/{query}"

# ---------------------------------------
# Flask routes
# ---------------------------------------
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/generate-playlist', methods=['POST'])
def generate_playlist():
    """
    Handle file uploads and generate playlist using recommender system
    """
    try:
        # Load artifacts
        artifacts = load_artifacts()
        if artifacts is None:
            return jsonify({'error': 'Recommender system not available. Please ensure model artifacts are present.'}), 500
        
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        all_recommendations = []
        saved_files = []
        
        # Process each uploaded file
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filepath)
                
                try:
                    # Extract audio features
                    features = extract_features_from_audio(filepath)
                    
                    # Map to model feature space
                    mapped_vec = smart_map_features_to_vector(features, artifacts["feature_cols"])
                    
                    # Get recommendations
                    recommendations = recommend_from_vector(mapped_vec.flatten(), artifacts, top_k=10)
                    all_recommendations.extend(recommendations)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # Clean up uploaded files
        for filepath in saved_files:
            try:
                os.remove(filepath)
            except:
                pass
        
        if not all_recommendations:
            return jsonify({'error': 'Could not generate recommendations from uploaded files'}), 400
        
        # Remove duplicates based on name and artist
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            key = (rec.get('name', ''), rec.get('artist', ''))
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        
        return jsonify({
            'success': True,
            'playlist': unique_recs[:20],  # Limit to top 20
            'message': f'Processed {len(saved_files)} audio files, found {len(unique_recs)} unique recommendations'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search-song', methods=['POST'])
def search_song():
    """
    Search for a song and get recommendations
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        # Load artifacts
        artifacts = load_artifacts()
        if artifacts is None:
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
            return jsonify({'error': f'No songs found matching "{query}"'}), 404
        
        # Get first match
        idx = matches.index[0]
        song_info = meta.iloc[idx].to_dict()
        
        # Get feature vector for this song
        vec = artifacts["features_df"][artifacts["feature_cols"]].iloc[idx].values
        
        # Get recommendations
        recommendations = recommend_from_vector(vec, artifacts, top_k=15)
        
        return jsonify({
            'success': True,
            'query_song': {
                'name': song_info.get('title', 'Unknown'),
                'artist': song_info.get('artist', 'Unknown Artist'),
                'genre': song_info.get('genre', 'Unknown')
            },
            'playlist': recommendations,
            'message': f'Found recommendations based on "{song_info.get("title")}" by {song_info.get("artist")}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-genres', methods=['GET'])
def get_genres():
    """
    Get list of available genres
    """
    try:
        artifacts = load_artifacts()
        if artifacts is None or artifacts["meta"] is None:
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
        data = request.get_json()
        genre = data.get('genre', '').strip()
        
        if not genre:
            return jsonify({'error': 'No genre provided'}), 400
        
        artifacts = load_artifacts()
        if artifacts is None:
            return jsonify({'error': 'Recommender system not available'}), 500
        
        meta = artifacts["meta"]
        df = artifacts["features_df"]
        fcols = artifacts["feature_cols"]
        
        # Find songs with this genre
        idxs = meta[meta["genre"] == genre].index.tolist()
        
        if not idxs:
            return jsonify({'error': f'Genre "{genre}" not found'}), 404
        
        # Calculate genre centroid
        centroid = np.mean(df[fcols].iloc[idxs].values, axis=0)
        
        # Get recommendations
        recommendations = recommend_from_vector(centroid, artifacts, top_k=15)
        
        return jsonify({
            'success': True,
            'genre': genre,
            'playlist': recommendations,
            'message': f'Generated playlist for {genre} genre'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-csv', methods=['POST'])
def download_csv():
    """
    Convert playlist JSON to CSV and send as download
    """
    try:
        data = request.get_json()
        playlist = data.get('playlist', [])
        
        if not playlist:
            return jsonify({'error': 'No playlist data provided'}), 400
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Track #', 'Name', 'Artist', 'Album', 'Genre', 'Duration', 'YouTube Link', 'Spotify Link'])
        
        # Write data
        for idx, track in enumerate(playlist, 1):
            writer.writerow([
                idx,
                track.get('name', ''),
                track.get('artist', ''),
                track.get('album', 'N/A'),
                track.get('genre', 'N/A'),
                track.get('duration', ''),
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
        return jsonify({'error': str(e)}), 500

@app.route('/download-json', methods=['POST'])
def download_json():
    """
    Send playlist as JSON file download
    """
    try:
        data = request.get_json()
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load artifacts on startup
    print("🎵 Starting Music Playlist Generator...")
    load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)