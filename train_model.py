"""
train_model.py — Train multi-trait AI model on FER2013 features.
Run:  python train_model.py

Pipeline:
  1. Download FER2013 via Kaggle
  2. Run MediaPipe on images → extract 10 features
  3. Compute heuristic scores for all 8 traits as training labels
  4. Train MultiOutputRegressor(RandomForest) → models/stress_ai.pkl
  5. Evaluate & save output/model_eval.png
"""
import os, csv, random, pathlib, time
import cv2, joblib, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from skillbrew.config import (
    FER_DIR, STRESS_MODEL_PATH, EMOTION_STRESS,
    MAX_TRAIN_IMAGES, OUTPUT_DIR, FEATURE_ORDER,
    ALL_TRAITS, TRAIT_CONFIGS,
)
from skillbrew.face_mesh_module    import FaceMeshProcessor
from skillbrew.feature_engineering import FeatureExtractor
from skillbrew.trait_analyzer      import BehavioralTraitAnalyzer

FEATURES_CSV = pathlib.Path("logs/fer_features.csv")
FEAT_COLS    = FEATURE_ORDER
TARGET_COLS  = ALL_TRAITS


# ── Step 1: Download FER2013 ──────────────────────────────────────
def download_fer2013():
    if FER_DIR.exists() and any(FER_DIR.rglob("*.png")):
        print(f"✅  FER2013 present at {FER_DIR}")
        return
    FER_DIR.mkdir(parents=True, exist_ok=True)
    print("⬇️   Downloading FER2013 from Kaggle...")
    ret = os.system(f'kaggle datasets download -d msambare/fer2013 --unzip -p "{FER_DIR}"')
    if ret != 0:
        raise RuntimeError(
            "Kaggle download failed.\n"
            "1. Set KAGGLE_USERNAME + KAGGLE_KEY in .env\n"
            "2. Run: python scripts/setup_kaggle.py"
        )
    print(f"✅  FER2013 downloaded.")


# ── Step 2: Extract features ──────────────────────────────────────
def extract_features() -> pathlib.Path:
    if FEATURES_CSV.exists() and FEATURES_CSV.stat().st_size > 1000:
        print(f"✅  Features CSV exists ({FEATURES_CSV}). Delete to re-extract.")
        return FEATURES_CSV

    split_dir = FER_DIR / "train"
    if not split_dir.exists():
        split_dir = FER_DIR

    all_images = []
    for emo_dir in sorted(split_dir.iterdir()):
        if not emo_dir.is_dir(): continue
        emo = emo_dir.name.lower()
        if emo not in EMOTION_STRESS: continue
        for p in list(emo_dir.glob("*.png")) + list(emo_dir.glob("*.jpg")):
            all_images.append((p, EMOTION_STRESS[emo]))

    if not all_images:
        raise FileNotFoundError(f"No images found under {split_dir}")

    random.seed(42); random.shuffle(all_images)
    if MAX_TRAIN_IMAGES > 0:
        all_images = all_images[:MAX_TRAIN_IMAGES]

    print(f"\n🖼️   Extracting features from {len(all_images)} images...")

    # We'll use the heuristic analyzer to generate all 8 trait labels
    heuristic_analyzer = BehavioralTraitAnalyzer()  # RULES mode (no model yet)

    FEATURES_CSV.parent.mkdir(exist_ok=True)
    all_cols = FEAT_COLS + TARGET_COLS
    processed = failed = 0

    with FaceMeshProcessor() as proc, open(FEATURES_CSV, "w", newline="") as f:
        writer    = csv.DictWriter(f, fieldnames=all_cols)
        extractor = FeatureExtractor()
        writer.writeheader()

        for img_path, stress_label in tqdm(all_images, desc="Extracting"):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None: failed += 1; continue

            up  = cv2.resize(gray, (192, 192), interpolation=cv2.INTER_CUBIC)
            bgr = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

            lm = proc.process(bgr)
            if lm is None: failed += 1; continue

            feats  = extractor.extract(lm)
            report = heuristic_analyzer.analyze(feats, timestamp=time.time())

            row = {**feats}
            for t in TARGET_COLS:
                row[t] = report.traits[t].score
            # Override stress with ground-truth FER label for better calibration
            row["stress"] = stress_label

            writer.writerow(row)
            f.flush()
            processed += 1

    print(f"✅  {processed} samples extracted ({failed} failed, "
          f"{processed/(processed+failed+1e-9)*100:.1f}% rate)")
    return FEATURES_CSV


# ── Step 3: Train ─────────────────────────────────────────────────
def train(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path).dropna()
    for col in FEAT_COLS:
        q = df[col].quantile(0.995)
        df = df[df[col] <= q]

    print(f"\n📊  Training on {len(df):,} samples  ×  {len(TARGET_COLS)} trait targets")

    X = df[FEAT_COLS].values
    Y = df[TARGET_COLS].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=4, max_features="sqrt",
                random_state=42, n_jobs=-1,
            ), n_jobs=-1
        ))
    ])

    print("🏋️   Training MultiOutput Random Forest pipeline...")
    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)
    print(f"\n📈  Per-trait performance (Test Set):")
    print(f"   {'Trait':<18}  {'MAE':>6}  {'R²':>6}")
    print(f"   {'─'*18}  {'─'*6}  {'─'*6}")
    for i, trait in enumerate(TARGET_COLS):
        mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        r2  = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"   {trait:<18}  {mae:>6.4f}  {r2:>6.4f}")

    STRESS_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, STRESS_MODEL_PATH)
    print(f"\n✅  Model saved → {STRESS_MODEL_PATH}")
    print(f"   Restart app.py to activate HYBRID mode.\n")

    _save_eval_plot(Y_test, Y_pred)


def _save_eval_plot(Y_test, Y_pred):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(TARGET_COLS)
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle("Skillbrew.AI — Multi-Trait Model Evaluation", fontsize=14, fontweight="bold")
        axes = axes.flat

        for i, trait in enumerate(TARGET_COLS):
            ax = axes[i]
            cfg = TRAIT_CONFIGS[trait]
            ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.3, s=5, color=cfg["color"])
            ax.plot([0,1],[0,1], "w--", lw=1, alpha=0.5)
            ax.set_title(cfg["label"], fontsize=9, color=cfg["color"])
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            r2 = r2_score(Y_test[:,i], Y_pred[:,i])
            ax.text(0.05, 0.92, f"R²={r2:.3f}", transform=ax.transAxes,
                    fontsize=8, color="white")
            ax.set_facecolor("#161b22"); ax.tick_params(colors="white", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#30363d")

        fig.patch.set_facecolor("#0d1117")
        plt.tight_layout()
        out = OUTPUT_DIR / "model_eval.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"📊  Eval plot saved → {out}")
    except Exception as e:
        print(f"   (Plot skipped: {e})")


if __name__ == "__main__":
    download_fer2013()
    csv_path = extract_features()
    train(csv_path)
