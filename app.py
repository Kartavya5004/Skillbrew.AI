"""
app.py — Skillbrew.AI v2  |  Fixed version
Fixes:
  1. socketio.start_background_task() instead of threading.Thread
  2. FaceMeshProcessor created inside the background task
  3. frame_only event so video streams even without a detected face
  4. REST fallback routes /api/camera/start and /api/camera/stop
  5. use_reloader=False to prevent double-init of background threads
"""
import base64
import time
import os

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from skillbrew.config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    CAMERA_INDEX, LOG_PATH, OUTPUT_DIR, FRONTEND_DIR,
    STRESS_MODEL_PATH, ALL_TRAITS, TRAIT_CONFIGS,
)
from skillbrew.face_mesh_module    import FaceMeshProcessor
from skillbrew.feature_engineering import FeatureExtractor
from skillbrew.trait_analyzer      import BehavioralTraitAnalyzer
from skillbrew.data_logger         import DataLogger

# ── App & SocketIO ────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)
app.config["SECRET_KEY"] = os.urandom(24)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ── Global state ──────────────────────────────────────────────────
_state = {
    "running":     False,
    "frame_count": 0,
    "camera_ok":   False,
    "model_mode":  "RULES",
    "fps":         0.0,
    "error":       None,
}
_should_stop = False

# ── Load ML model once at startup ────────────────────────────────
print("Loading AI model...")
_analyzer  = BehavioralTraitAnalyzer(model_path=STRESS_MODEL_PATH)
_extractor = FeatureExtractor()
_state["model_mode"] = _analyzer.method
print(f"Model mode: {_state['model_mode']}")


def _draw_frame(bgr, report):
    h, w = bgr.shape[:2]
    ov = bgr.copy()
    cv2.rectangle(ov, (0, 0), (w, 88), (10, 12, 18), -1)
    cv2.addWeighted(ov, 0.72, bgr, 0.28, 0, bgr)
    cv2.putText(bgr, "Skillbrew.AI", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 1, cv2.LINE_AA)
    sorted_t = sorted(report.traits.values(), key=lambda x: x.score, reverse=True)
    x = 12
    for ts in sorted_t[:4]:
        lbl = f"{ts.icon} {ts.name[:6]}:{ts.score:.2f}"
        cv2.putText(bgr, lbl, (x, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200, 210, 255), 1, cv2.LINE_AA)
        x += 150
    bw = int(report.overall_score * (w - 24))
    cv2.rectangle(bgr, (12, 65), (12 + bw, 74), (80, 160, 255), -1)
    cv2.rectangle(bgr, (12, 65), (w - 12, 74), (60, 70, 90), 1)
    return bgr


def _camera_task():
    global _should_stop
    _state["error"] = None

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        msg = f"Cannot open camera {CAMERA_INDEX}. Change CAMERA_INDEX in .env"
        _state["running"] = False
        _state["camera_ok"] = False
        _state["error"] = msg
        socketio.emit("camera_error", {"message": msg})
        print(f"[Camera] ERROR: {msg}")
        return

    _state["camera_ok"] = True
    print(f"[Camera] Opened index {CAMERA_INDEX} OK")

    try:
        proc = FaceMeshProcessor()
    except Exception as e:
        msg = f"MediaPipe init failed: {e}"
        _state["running"] = False
        _state["camera_ok"] = False
        _state["error"] = msg
        socketio.emit("camera_error", {"message": msg})
        cap.release()
        return

    logger    = DataLogger(LOG_PATH)
    t_fps     = time.time()
    fps_count = 0

    try:
        while not _should_stop:
            ok, bgr = cap.read()
            if not ok:
                socketio.sleep(0.03)
                continue

            lm = proc.process(bgr)

            if lm is None:
                # Emit raw frame even without face detection
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buf).decode()
                socketio.emit("frame_only", {"frame_b64": b64})
                socketio.sleep(0.001)
                continue

            feats  = _extractor.extract(lm)
            report = _analyzer.analyze(feats, timestamp=time.time())
            logger.log(feats, report)

            _state["frame_count"] += 1
            fps_count             += 1
            now = time.time()
            if now - t_fps >= 1.0:
                _state["fps"] = fps_count / (now - t_fps)
                fps_count = 0
                t_fps     = now

            annotated = _draw_frame(bgr.copy(), report)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
            b64 = base64.b64encode(buf).decode()

            payload             = report.to_dict()
            payload["frame_b64"] = b64
            payload["fps"]       = round(_state["fps"], 1)
            socketio.emit("trait_update", payload)
            socketio.sleep(0.001)

    except Exception as e:
        _state["error"] = str(e)
        socketio.emit("camera_error", {"message": str(e)})
        print(f"[Camera] Error: {e}")

    finally:
        proc.close()
        logger.close()
        cap.release()
        _state["running"]   = False
        _state["camera_ok"] = False
        _should_stop        = False
        socketio.emit("status_update", {"running": False, "camera_ok": False})
        print("[Camera] Stopped")


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "running":     _state["running"],
        "camera_ok":   _state["camera_ok"],
        "model_mode":  _state["model_mode"],
        "frame_count": _state["frame_count"],
        "fps":         _state["fps"],
        "error":       _state["error"],
        "traits":      ALL_TRAITS,
        "trait_meta": {
            k: {"label": v["label"], "icon": v["icon"], "color": v["color"],
                "description": v["description"], "high_is_bad": v["high_is_bad"]}
            for k, v in TRAIT_CONFIGS.items()
        },
    })


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    global _should_stop
    if _state["running"]:
        return jsonify({"ok": True, "message": "Already running"})
    _should_stop          = False
    _state["running"]     = True
    _state["frame_count"] = 0
    socketio.start_background_task(_camera_task)
    return jsonify({"ok": True})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    global _should_stop
    _should_stop      = True
    _state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/session/history")
def api_history():
    try:
        import pandas as pd
        if not LOG_PATH.exists() or LOG_PATH.stat().st_size < 100:
            return jsonify({"rows": [], "count": 0})
        df = pd.read_csv(LOG_PATH).tail(300)
        return jsonify({"rows": df.to_dict(orient="records"), "count": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/reset", methods=["POST"])
def api_reset():
    try:
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/generate", methods=["POST"])
def api_generate_report():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
        import pandas as pd

        if not LOG_PATH.exists() or LOG_PATH.stat().st_size < 100:
            return jsonify({"error": "No session data. Run camera first."}), 400
        df = pd.read_csv(LOG_PATH).dropna()
        if len(df) < 5:
            return jsonify({"error": f"Only {len(df)} frames. Need at least 5."}), 400

        trait_cols = [t for t in ALL_TRAITS if t in df.columns]
        n_cols = 4
        n_rows = -(-len(trait_cols) // n_cols)
        fig = plt.figure(figsize=(20, 5 * n_rows))
        fig.suptitle("Skillbrew.AI — Behavioral Analytics Report",
                     fontsize=16, color="white", fontweight="bold")
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.5, wspace=0.35)
        for i, trait in enumerate(trait_cols):
            ax    = fig.add_subplot(gs[i // n_cols, i % n_cols])
            color = TRAIT_CONFIGS[trait]["color"]
            roll  = df[trait].rolling(5, min_periods=1).mean()
            ax.fill_between(range(len(roll)), roll, alpha=0.2, color=color)
            ax.plot(roll, color=color, lw=1.8)
            ax.set_ylim(0, 1)
            ax.set_title(TRAIT_CONFIGS[trait]["label"], color=color, fontsize=9)
            ax.tick_params(labelsize=7, colors="white")
        fig.patch.set_facecolor("#0d1117")
        plt.tight_layout()
        out = OUTPUT_DIR / "report.png"
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close()

        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return jsonify({"image_b64": b64})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── SocketIO events ───────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print(f"[WS] connected {request.sid}")
    emit("status_update", {
        "running":    _state["running"],
        "camera_ok":  _state["camera_ok"],
        "model_mode": _state["model_mode"],
    })


@socketio.on("disconnect")
def on_disconnect():
    print(f"[WS] disconnected {request.sid}")


@socketio.on("start_camera")
def on_start():
    global _should_stop
    print("[WS] start_camera")
    if _state["running"]:
        emit("status_update", {"running": True, "model_mode": _state["model_mode"]})
        return
    _should_stop          = False
    _state["running"]     = True
    _state["frame_count"] = 0
    socketio.start_background_task(_camera_task)
    emit("status_update", {"running": True, "model_mode": _state["model_mode"]})


@socketio.on("stop_camera")
def on_stop():
    global _should_stop
    print("[WS] stop_camera")
    _should_stop      = True
    _state["running"] = False
    emit("status_update", {"running": False})


# ── Entry ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🎯 Skillbrew.AI v2 → http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   Model : {_state['model_mode']}  |  Camera : {CAMERA_INDEX}\n")
    socketio.run(
        app,
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
