# 🎯 Skillbrew.AI — Interview Behavioral Analytics
### AI-PS-4 | Hackathon Submission

> Real-time multimodal behavioral trait extraction from interview video.  
> 8 traits • Explainable AI • Flask + WebSocket frontend • FER2013-trained hybrid model

---

## 🧠 What It Does

Skillbrew.AI processes live webcam video to extract **10 facial features** and scores **8 behavioral traits** in real-time. Each trait has heuristic weights validated against the FER2013 emotion dataset, blended with a Random Forest model (HYBRID mode after training).

### 8 Behavioral Traits Measured

| Trait | Signal Source | Relevance |
|-------|--------------|-----------|
| 🔴 **Stress Level**     | Jaw tension, brow furrow, blink rate | Interview anxiety |
| 💪 **Confidence**       | Gaze directness, head stability | Candidate assurance |
| ✨ **Engagement**       | Nodding, eye contact, brow activity | Active listening |
| 🎯 **Focus**            | Gaze concentration, minimal movement | Attention quality |
| ⚠️ **Deception Risk**   | Gaze aversion, micro-expressions | Authenticity signal |
| 🧠 **Cognitive Load**   | Upward gaze, blink increase, brow furrow | Mental effort |
| 😊 **Positivity**       | Smile signals, relaxed brow | Candidate demeanour |
| 😬 **Nervousness**      | Rapid micro-movements, lip compression | Anxiety level |

### 10 Facial Features Extracted

`eyebrow_raise` · `lip_tension` · `nod` · `blink_rate` · `jaw_tension` · `head_tilt_stability` · `gaze_directness` · `mouth_activity` · `micro_expression_score` · `facial_asymmetry`

---

## Project Structure

```
Skillbrew.AI/
├── app.py                    ← Flask + SocketIO API server  (ENTRY POINT)
├── train_model.py            ← Download FER2013 + train multi-trait RF
├── requirements.txt
├── .env.example
│
├── skillbrew/                ← Core Python package
│   ├── config.py             ← All config, trait definitions, weights
│   ├── face_mesh_module.py   ← MediaPipe FaceLandmarker (478 points)
│   ├── feature_engineering.py ← 10-feature extractor
│   ├── trait_analyzer.py     ← 8-trait scorer + XAI evidence engine
│   ├── data_logger.py        ← CSV session logger
│   └── face_landmarker.task  ← MediaPipe model (downloaded by script)
│
├── frontend/
│   ├── templates/index.html  ← Dashboard HTML
│   └── static/
│       ├── css/dashboard.css ← Dark analytical UI
│       └── js/dashboard.js   ← WebSocket + Chart.js live rendering
│
├── scripts/
│   ├── download_model.py     ← Downloads face_landmarker.task
│   └── setup_kaggle.py       ← Writes ~/.kaggle/kaggle.json from .env
│
├── .vscode/launch.json       ← F5 run configs
└── install.bat               ← Windows one-click setup
```

---

##  Quick Start (Windows)

### Step 1 — Install everything
```
Double-click  install.bat
```
This creates the venv, installs all packages, downloads MediaPipe model.

### Step 2 — Configure
Edit `.env`:
```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key        # kaggle.com → Account → API → Create New Token
```

### Step 3 — Train the AI model *(optional but recommended)*
```powershell
.venv\Scripts\activate
python scripts/setup_kaggle.py   # writes ~/.kaggle/kaggle.json
python train_model.py            # ~5–10 min, downloads FER2013 ~60MB
```

### Step 4 — Launch
```powershell
python app.py
```
Open **Link in Terminal** in your browser → click **▶ Start**.

---

##  Architecture

```
Browser (WebSocket) ←──────────────────────────────────────────┐
       ↓                                                        │
   Flask + SocketIO  app.py                                     │
       ↓                                                        │
   Camera Thread (background)                                   │
       ↓                                                        │
   FaceMeshProcessor          ← MediaPipe 478 landmarks         │
       ↓                                                        │
   FeatureExtractor           ← 10 normalised features          │
       ↓                                                        │
   BehavioralTraitAnalyzer                                      │
    ├─ Heuristic scoring  (always active)                       │
    └─ RF AI scoring      (active after training)               │
           ↓                                                    │
     BehavioralReport  ─── DataLogger (CSV) ───────────────────┘
           ↓
     SocketIO emit("trait_update", report)
           ↓
     Dashboard JS → Chart.js timeline, radar, evidence panel
```

### Hybrid Model Logic
```
Final Score = 0.70 × RandomForest(features) + 0.30 × Heuristic(features)
```
- **Before training**: RULES mode (pure heuristic, works immediately)
- **After training**: HYBRID mode (FER2013-calibrated RF + rules)

### Explainability (XAI)
Each trait produces evidence strings like:
> *"Gaze aversion pattern detected (0.312) — potential discomfort signal"*
> *"Elevated blink rate (0.341) — classic anxiety marker"*

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Dashboard frontend |
| GET  | `/api/status` | System status + trait metadata |
| GET  | `/api/session/history` | Last 300 logged frames |
| POST | `/api/session/reset` | Clear session log |
| POST | `/api/analyze/frame` | Analyze base64 image (REST) |
| POST | `/api/report/generate` | Generate PNG report |

**WebSocket events** (server → client):
- `trait_update` — full BehavioralReport per frame
- `status_update` — camera/model status change

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `mediapipe` not found | `pip install mediapipe>=0.10.30` |
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'flask_socketio'` | `pip install flask-socketio eventlet` |
| `face_landmarker.task not found` | `python scripts/download_model.py` |
| Camera won't open | Change `CAMERA_INDEX` in `.env` (try 1, 2) |
| Kaggle download fails | Run `python scripts/setup_kaggle.py` first |
| Port 5000 in use | Change `FLASK_PORT=5001` in `.env` |
