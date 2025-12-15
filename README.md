# 🎧 Noctune – AI-Powered Music Playlist Generator

Noctune is an AI-powered music recommendation and playlist generation system that analyzes audio characteristics to suggest songs with similar musical profiles. It uses machine learning and audio signal processing to create personalized playlists through a simple web interface.

---

## 📖 Table of Contents

* Project Overview
* Features
* Tech Stack
* Project Structure
* How It Works
* Requirements
* Installation & Setup
* Running the Application
* Usage Guide
* Dataset & Models
* Troubleshooting
* Future Improvements

---

## 📌 Project Overview

Noctune allows users to upload an audio file and receive a playlist of musically similar songs. The system extracts audio features from the uploaded track and compares them against a large music dataset to generate recommendations.

---

## ✨ Features

* AI-based music recommendations using K-Nearest Neighbors (KNN)
* Audio feature extraction using librosa (MFCC, chroma, tempo, spectral features)
* Supports MP3, WAV, FLAC, M4A, and OGG formats
* Pre-trained ML models for fast recommendations
* Flask-based REST API backend
* Modern, responsive web interface

---

## 🛠 Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** scikit-learn, NumPy, Pandas
* **Audio Processing:** librosa, soundfile
* **Frontend:** HTML, CSS, JavaScript
* **Model Persistence:** joblib

---

## 📂 Project Structure

```
Noctune-AI-Powered_Music-Playlist-Generator/
│── app.py / app1.py          # Flask application
│── models/
│   ├── nn.joblib             # Pre-trained recommendation model
│   ├── scaler.joblib         # Feature scaler
│   └── feature_cols.json     # Feature column mapping
│── data/
│   ├── fma_song_metadata.csv
│   └── fma_features_processed.csv
│── static/                   # CSS, JS, assets
│── templates/                # HTML files
│── main.ipynb
README.md
```

---

## ⚙️ How It Works

1. User uploads an audio file through the web interface
2. Audio features are extracted using librosa
3. Features are normalized using a pre-trained scaler
4. KNN model finds similar songs from the FMA dataset
5. A playlist of recommended songs is displayed to the user

---

## 📦 Requirements

Make sure you have **Python 3.8 or above** installed.

### Python Dependencies

```
flask==2.3.0
flask-cors==4.0.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
joblib==1.2.0
librosa==0.10.0
soundfile==0.12.0
werkzeug==2.3.0
```

---

## 🔧 Installation & Setup

### Step 1: Clone the Repository

```
git clone https://github.com/anandsarode22/Noctune-AI-Powered-Music-Playlist-Generator.git
cd noctune
```

### Step 2: Create a Virtual Environment (Recommended)

```
python -m venv venv
```

Activate the virtual environment:

* **Linux / macOS**

```
source venv/bin/activate
```

* **Windows**

```
venv\Scripts\activate
```

### Step 3: Install Dependencies

You can install all required packages using:

```
pip install flask flask-cors numpy pandas scikit-learn joblib librosa soundfile werkzeug
```

Or using `requirements.txt`:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Flask server using either file:

```
python app.py
```

or

```
python app1.py
```

Once running, open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🧑‍💻 Usage Guide

1. Open the Noctune web app in your browser
2. Upload a supported audio file
3. Click on **Generate Playlist**
4. View the list of recommended songs

---

## 📊 Dataset & Models

* **Dataset:** Free Music Archive (FMA)
* **Metadata File:** `fma_song_metadata.csv`
* **Audio Features File:** `fma_features_processed.csv`
* **Model:** Pre-trained KNN-based recommendation model

---

## 🐞 Troubleshooting

* Ensure all dependencies are installed correctly
* Check Python version compatibility
* Make sure model and data files are in the correct directories
* If librosa fails, verify that soundfile is installed properly

---

## 🚀 Future Improvements

* User authentication and saved playlists
* Real-time streaming integration
* Deep learning-based recommendation models
* Improved UI/UX

---

## 📜 License

This project is for educational and demonstration purposes.

---

## 🙌 Acknowledgements

* Free Music Archive (FMA)
* librosa and scikit-learn open-source communities
