"""
Train and evaluate the perfume recommendation ML model.
Uses a Random Forest classifier with cross-validation and saves the model.

Run: python train_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection  import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors        import KNeighborsClassifier

# ── Import data generator ────────────────────────────────────────────────────
from generate_data import generate_dataset, PERFUMES, MOODS, SEASONS, OCCASIONS

FEATURE_COLS = ["energy","nature","romance","adventurous","classic","warmth","bold",
                "mood_enc","season_enc","occasion_enc"]
LABEL_COL    = "label"

PERFUME_META = {p["id"]: {k: v for k, v in p.items() if k != "profile"} for p in PERFUMES}


def load_or_generate(path="perfume_data.csv", n=6000):
    if Path(path).exists():
        print(f"  Loading existing dataset from {path}")
        return pd.read_csv(path)
    print(f"  Generating {n} training samples…")
    df = generate_dataset(n)
    df.to_csv(path, index=False)
    return df


def train():
    print("\n" + "═"*58)
    print("   SCENTOIRE · ML Model Training")
    print("═"*58)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data…")
    df = load_or_generate(n=6000)
    X  = df[FEATURE_COLS].values
    y  = df[LABEL_COL].values
    print(f"  ✓ {len(df)} samples, {len(PERFUMES)} classes")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting train/test (80/20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Models to compare ────────────────────────────────────────────────────
    print("\n[3/5] Comparing models…")
    candidates = {
        "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                      min_samples_split=4, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                          max_depth=5, random_state=42),
        "KNN":                KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }

    best_name, best_score, best_model = None, 0, None
    for name, clf in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  {name:22s}  CV acc: {cv.mean():.3f} ± {cv.std():.3f}")
        if cv.mean() > best_score:
            best_score = cv.mean(); best_name = name; best_model = pipe

    print(f"\n  ★ Best model: {best_name}  ({best_score:.3f})")

    # ── Fine-tune RF ──────────────────────────────────────────────────────────
    print("\n[4/5] Fine-tuning Random Forest…")
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_split=3,
            min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    final_pipe.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set…")
    y_pred = final_pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")

    label_names = [PERFUME_META[i]["name"] for i in sorted(PERFUME_META)]
    print("\n  Per-class Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # ── Feature importance ────────────────────────────────────────────────────
    rf         = final_pipe.named_steps["clf"]
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
    print("  Feature Importances:")
    for f, v in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 60)
        print(f"    {f:15s} {bar}  {v:.4f}")

    # ── Save model + metadata ─────────────────────────────────────────────────
    print("\n  Saving model…")
    joblib.dump(final_pipe, "perfume_model.pkl")

    meta = {
        "feature_cols": FEATURE_COLS,
        "moods":        MOODS,
        "seasons":      SEASONS,
        "occasions":    OCCASIONS,
        "perfumes":     {str(k): v for k, v in PERFUME_META.items()},
        "accuracy":     round(acc, 4),
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("  ✓ perfume_model.pkl  saved")
    print("  ✓ model_meta.json    saved")
    print("\n" + "═"*58)
    print(f"  Training complete!  Accuracy: {acc*100:.1f}%")
    print("═"*58 + "\n")
    return final_pipe, meta


if __name__ == "__main__":
    train()
