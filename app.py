"""
Flask REST API — Perfume Recommendation Backend
Serves ML predictions to the HTML frontend.

Run:
    python app.py

Endpoints:
    POST /recommend   { answers: {...} }  →  { recommendations, personality_summary }
    GET  /health                          →  { status, model_accuracy }
    GET  /perfumes                        →  full perfume list
"""
import json
import warnings
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, make_response

# ── Load model (auto-retrain on sklearn version mismatch) ─────────────────────
MODEL_PATH = Path("perfume_model.pkl")
META_PATH  = Path("model_meta.json")


def load_model():
    """Load model, auto-retrain locally if pkl was built on a different sklearn."""
    if not MODEL_PATH.exists():
        print("⚠  Model not found — training now…")
        from train_model import train
        train()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = joblib.load(MODEL_PATH)
        has_mismatch = any(
            "InconsistentVersionWarning" in str(w.category) for w in caught
        )

    if has_mismatch:
        print("⚠  sklearn version mismatch — retraining on your environment…")
        from train_model import train
        train()
        m = joblib.load(MODEL_PATH)
        print("✓  Model retrained successfully!")
    return m


model = load_model()

with open(META_PATH) as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
MOODS        = meta["moods"]
SEASONS      = meta["seasons"]
OCCASIONS    = meta["occasions"]
PERFUMES     = meta["perfumes"]   # keyed by string id "0"…"11"

print(f"✓ Model loaded  (accuracy: {meta['accuracy']*100:.1f}%)")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/recommend", methods=["OPTIONS"])
@app.route("/health",    methods=["OPTIONS"])
@app.route("/perfumes",  methods=["OPTIONS"])
def options():
    return make_response("", 204)


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_answers(answers: dict) -> np.ndarray:
    mood_enc     = MOODS.index(answers["mood"])         if answers.get("mood")     in MOODS     else 0
    season_enc   = SEASONS.index(answers["season"])     if answers.get("season")   in SEASONS   else 0
    occasion_enc = OCCASIONS.index(answers["occasion"]) if answers.get("occasion") in OCCASIONS else 0
    row = [
        int(answers.get("energy",      3)),
        int(answers.get("nature",      3)),
        int(answers.get("romance",     3)),
        int(answers.get("adventurous", 3)),
        int(answers.get("classic",     3)),
        int(answers.get("warmth",      3)),
        int(answers.get("bold",        3)),
        mood_enc, season_enc, occasion_enc,
    ]
    return np.array(row).reshape(1, -1)


def compute_match_score(proba: float, rank: int) -> int:
    base    = int(proba * 100)
    penalty = rank * 4
    return max(60, min(99, base - penalty + (5 if rank == 0 else 0)))


def build_personality_summary(answers: dict) -> str:
    energy  = int(answers.get("energy",  3))
    romance = int(answers.get("romance", 3))
    bold    = int(answers.get("bold",    3))
    nature  = int(answers.get("nature",  3))
    mood    = answers.get("mood",   "")
    season  = answers.get("season", "")

    soul     = "introspective" if energy  >= 4 else "vibrant"          if energy  <= 2 else "balanced"
    heart    = "deeply romantic" if romance >= 4 else "pragmatic"      if romance <= 2 else "warmly feeling"
    presence = "bold and magnetic" if bold >= 4 else "quietly compelling" if bold <= 2 else "elegantly understated"
    world    = "grounded in nature" if nature >= 4 else "drawn to urban sophistication" if nature <= 2 else "equally at home in city and forest"

    season_word = season.split("–")[0].strip() if "–" in season else season
    mood_word   = mood.split("&")[0].strip()   if "&" in mood   else mood

    return (
        f"You carry a {soul}, {heart} soul — {presence} and {world}. "
        f"With a {mood_word.lower()} spirit and a natural affinity for {season_word.lower()}, "
        f"your signature scent should feel like a second skin: intimate, memorable, and unmistakably yours."
    )


# ── Copy-writing ──────────────────────────────────────────────────────────────
_HEADLINES = {
    "Chanel No. 5":       "Timeless florals for the eternal romantic",
    "Black Opium":        "Dark, addictive, unmistakably bold",
    "Sauvage":            "Raw freedom meets polished edge",
    "Flowerbomb":         "Where warmth blooms into wonder",
    "Terre d'Hermès":     "Earth, cedar, and sunlit solitude",
    "Oud Wood":           "Smoldering luxury for the daring soul",
    "Light Blue":         "A Mediterranean breeze in a bottle",
    "Shalimar":           "Oriental legend for timeless romantics",
    "Acqua di Gio":       "Coastal clarity, effortless allure",
    "Tobacco Vanille":    "Autumnal richness, cozy complexity",
    "Coco Mademoiselle":  "Spirited elegance, modern femininity",
    "Bleu de Chanel":     "Refined freedom, self-defined style",
}

_JOURNEYS = {
    "Chanel No. 5":       "Opens with sparkling aldehydes, blooms into powdery rose, and dries down to a warm, creamy musk that lingers like a cherished memory.",
    "Black Opium":        "A rush of roasted coffee gives way to white florals, melting into a sweet vanilla and cedarwood embrace that pulses through the night.",
    "Sauvage":            "Bursts with bergamot and cracked pepper, settling into a vast, wind-swept ambroxan that evokes endless open terrain.",
    "Flowerbomb":         "Tea and citrus ignite the senses before a cascade of rose and jasmine bloom, resting on a warm patchouli and musk foundation.",
    "Terre d'Hermès":     "Grapefruit sparks against flint, revealing cool vetiver and cedar that deepen into a quietly smoldering benzoin.",
    "Oud Wood":           "Cardamom and rosewood kindle the opening, as rare oud weaves through sandalwood into a rich, resinous amber finale.",
    "Light Blue":         "Crisp Sicilian lemon and green apple refresh the senses, fading gently into bamboo and cedar with a whisper of white musk.",
    "Shalimar":           "Bergamot illuminates an iris and rose heart before the legendary vanilla, oud, and civet draw you into oriental reverie.",
    "Acqua di Gio":       "Marine minerality and citrus open to neroli and rosemary, drying down to a clean cedarwood and patchouli accord that breathes like sea air.",
    "Tobacco Vanille":    "Spiced tobacco and dried fruit unfurl into a sumptuous tonka and vanilla heart, anchored by warming wood sap.",
    "Coco Mademoiselle":  "Orange and bergamot spark a vibrant opening, while rose and jasmine soften into a sensual patchouli and vetiver trail.",
    "Bleu de Chanel":     "Citrus and ginger lift the spirits before incense adds gravitas, settling into sandalwood and cedar — effortlessly clean and endlessly refined.",
}


def _personality_match(p, answers):
    energy  = int(answers.get("energy",  3))
    romance = int(answers.get("romance", 3))
    bold    = int(answers.get("bold",    3))
    soul    = "reflective nature" if energy >= 4 else "lively spirit" if energy <= 2 else "balanced character"
    anchor  = p["accords"][0].lower() if p["accords"] else "refined"
    trait   = "romantic depth" if romance >= 4 else "confident presence" if bold >= 4 else "grounded authenticity"
    return (
        f"Your {soul} aligns beautifully with {p['name']}'s {anchor} character — "
        f"a fragrance that speaks without words. "
        f"The blend mirrors your {trait}, making it feel less like a choice "
        f"and more like a natural extension of who you are."
    )


def _key_reason(p, answers):
    mood     = answers.get("mood",     "")
    occasion = answers.get("occasion", "")
    accord   = p["accords"][0] if p["accords"] else "unique"
    return f"Its {accord.lower()} character perfectly suits a {mood.lower()} mood for {occasion.lower()}."


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    data    = request.get_json(force=True)
    answers = data.get("answers", {})
    if not answers:
        return jsonify({"error": "No answers provided"}), 400

    X         = encode_answers(answers)
    proba_all = model.predict_proba(X)[0]
    classes   = model.classes_

    top3_idx   = np.argsort(proba_all)[::-1][:3]
    top3_ids   = [str(classes[i]) for i in top3_idx]
    top3_proba = [proba_all[i]    for i in top3_idx]

    recommendations = []
    for rank, (pid, prob) in enumerate(zip(top3_ids, top3_proba)):
        p   = PERFUMES[pid]
        rec = {
            "name":              p["name"],
            "brand":             p["brand"],
            "notes":             p["notes"],
            "accords":           p["accords"],
            "rating":            p["rating"],
            "icon":              p["icon"],
            "match_score":       compute_match_score(prob, rank),
            "headline":          _HEADLINES.get(p["name"], "A signature scent for your spirit"),
            "personality_match": _personality_match(p, answers),
            "scent_journey":     _JOURNEYS.get(p["name"], "A beautifully layered fragrance that unfolds over time."),
            "key_reason":        _key_reason(p, answers),
        }
        recommendations.append(rec)

    return jsonify({
        "recommendations":     recommendations,
        "personality_summary": build_personality_summary(answers),
        "model_confidence":    round(float(top3_proba[0]), 3),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model":         "RandomForest",
        "accuracy":      f"{meta['accuracy']*100:.1f}%",
        "perfume_count": len(PERFUMES),
        "feature_count": len(FEATURE_COLS),
    })


@app.route("/perfumes", methods=["GET"])
def get_perfumes():
    return jsonify(list(PERFUMES.values()))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌹 Scentoire ML Backend")
    print("   http://localhost:8001/recommend  (POST)")
    print("   http://localhost:8001/health     (GET)")
    print("   http://localhost:8001/perfumes   (GET)\n")
    app.run(debug=True, port=8001)
