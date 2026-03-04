"""
Generate synthetic training data for the perfume recommendation ML model.
Each sample = personality profile (10 features) + best matching perfume label.
"""
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

PERFUMES = [
    {
        "id": 0, "name": "Chanel No. 5", "brand": "Chanel",
        "notes": "Aldehydes, Ylang-ylang, Rose, Jasmine, Sandalwood, Vanilla",
        "accords": ["Floral", "Powdery", "Aldehyde"], "rating": 4.8, "icon": "🌹",
        # Ideal personality profile (energy, nature, romance, adventurous, classic, warmth, bold)
        "profile": {"energy": 4, "nature": 2, "romance": 5, "adventurous": 2, "classic": 5, "warmth": 4, "bold": 3},
        "mood_fit": ["Romantic & Dreamy"], "season_fit": ["Spring – Fresh & Hopeful", "Winter – Cool & Refined"],
        "occasion_fit": ["Special occasions", "Evening & nightlife"],
    },
    {
        "id": 1, "name": "Black Opium", "brand": "YSL",
        "notes": "Coffee, Vanilla, White flowers, Cedarwood",
        "accords": ["Sweet", "Gourmand", "Floral"], "rating": 4.6, "icon": "☕",
        "profile": {"energy": 2, "nature": 1, "romance": 3, "adventurous": 4, "classic": 2, "warmth": 3, "bold": 5},
        "mood_fit": ["Mysterious & Intense", "Energized & Upbeat"],
        "season_fit": ["Autumn – Rich & Complex", "Winter – Cool & Refined"],
        "occasion_fit": ["Evening & nightlife", "Special occasions"],
    },
    {
        "id": 2, "name": "Sauvage", "brand": "Dior",
        "notes": "Bergamot, Pepper, Lavender, Ambroxan, Cedar",
        "accords": ["Fresh", "Aromatic", "Spicy"], "rating": 4.7, "icon": "🌊",
        "profile": {"energy": 2, "nature": 4, "romance": 2, "adventurous": 5, "classic": 3, "warmth": 2, "bold": 5},
        "mood_fit": ["Energized & Upbeat", "Mysterious & Intense"],
        "season_fit": ["Summer – Warm & Vibrant", "Spring – Fresh & Hopeful"],
        "occasion_fit": ["Daily wear", "Work & professional"],
    },
    {
        "id": 3, "name": "Flowerbomb", "brand": "Viktor & Rolf",
        "notes": "Bergamot, Tea, Jasmine, Rose, Patchouli, Musk",
        "accords": ["Floral", "Sweet", "Powdery"], "rating": 4.5, "icon": "💐",
        "profile": {"energy": 3, "nature": 3, "romance": 5, "adventurous": 3, "classic": 3, "warmth": 5, "bold": 3},
        "mood_fit": ["Romantic & Dreamy", "Energized & Upbeat"],
        "season_fit": ["Spring – Fresh & Hopeful", "Summer – Warm & Vibrant"],
        "occasion_fit": ["Special occasions", "Daily wear"],
    },
    {
        "id": 4, "name": "Terre d'Hermès", "brand": "Hermès",
        "notes": "Grapefruit, Orange, Flint, Vetiver, Cedar, Benzoin",
        "accords": ["Woody", "Earthy", "Citrus"], "rating": 4.6, "icon": "🌿",
        "profile": {"energy": 4, "nature": 5, "romance": 2, "adventurous": 3, "classic": 4, "warmth": 3, "bold": 2},
        "mood_fit": ["Calm & Reflective"],
        "season_fit": ["Autumn – Rich & Complex", "Summer – Warm & Vibrant"],
        "occasion_fit": ["Daily wear", "Work & professional"],
    },
    {
        "id": 5, "name": "Oud Wood", "brand": "Tom Ford",
        "notes": "Oud, Rosewood, Cardamom, Sandalwood, Amber",
        "accords": ["Woody", "Oud", "Spicy"], "rating": 4.7, "icon": "🪵",
        "profile": {"energy": 3, "nature": 2, "romance": 4, "adventurous": 4, "classic": 4, "warmth": 3, "bold": 5},
        "mood_fit": ["Mysterious & Intense"],
        "season_fit": ["Autumn – Rich & Complex", "Winter – Cool & Refined"],
        "occasion_fit": ["Evening & nightlife", "Special occasions"],
    },
    {
        "id": 6, "name": "Light Blue", "brand": "Dolce & Gabbana",
        "notes": "Sicilian lemon, Apple, Cedar, Bamboo, White rose, Musk",
        "accords": ["Fresh", "Aquatic", "Citrus"], "rating": 4.4, "icon": "🍋",
        "profile": {"energy": 2, "nature": 4, "romance": 3, "adventurous": 3, "classic": 2, "warmth": 4, "bold": 2},
        "mood_fit": ["Energized & Upbeat", "Calm & Reflective"],
        "season_fit": ["Summer – Warm & Vibrant", "Spring – Fresh & Hopeful"],
        "occasion_fit": ["Daily wear", "Work & professional"],
    },
    {
        "id": 7, "name": "Shalimar", "brand": "Guerlain",
        "notes": "Bergamot, Iris, Rose, Jasmine, Vanilla, Oud, Civet",
        "accords": ["Oriental", "Powdery", "Vanilla"], "rating": 4.8, "icon": "✨",
        "profile": {"energy": 4, "nature": 2, "romance": 5, "adventurous": 3, "classic": 5, "warmth": 4, "bold": 4},
        "mood_fit": ["Romantic & Dreamy", "Mysterious & Intense"],
        "season_fit": ["Winter – Cool & Refined", "Autumn – Rich & Complex"],
        "occasion_fit": ["Special occasions", "Evening & nightlife"],
    },
    {
        "id": 8, "name": "Acqua di Gio", "brand": "Armani",
        "notes": "Marine, Bergamot, Neroli, Rosemary, Cedarwood, Patchouli",
        "accords": ["Fresh", "Aquatic", "Aromatic"], "rating": 4.6, "icon": "🌬️",
        "profile": {"energy": 2, "nature": 5, "romance": 2, "adventurous": 4, "classic": 3, "warmth": 3, "bold": 2},
        "mood_fit": ["Energized & Upbeat", "Calm & Reflective"],
        "season_fit": ["Summer – Warm & Vibrant", "Spring – Fresh & Hopeful"],
        "occasion_fit": ["Daily wear", "Work & professional"],
    },
    {
        "id": 9, "name": "Tobacco Vanille", "brand": "Tom Ford",
        "notes": "Tobacco leaf, Vanilla, Tonka bean, Dried fruits, Wood sap",
        "accords": ["Warm Spicy", "Sweet", "Vanilla"], "rating": 4.8, "icon": "🍂",
        "profile": {"energy": 4, "nature": 3, "romance": 3, "adventurous": 3, "classic": 4, "warmth": 4, "bold": 4},
        "mood_fit": ["Calm & Reflective", "Mysterious & Intense"],
        "season_fit": ["Autumn – Rich & Complex", "Winter – Cool & Refined"],
        "occasion_fit": ["Evening & nightlife", "Special occasions"],
    },
    {
        "id": 10, "name": "Coco Mademoiselle", "brand": "Chanel",
        "notes": "Orange, Bergamot, Rose, Jasmine, Patchouli, Vetiver, Musk",
        "accords": ["Floral", "Woody", "Citrus"], "rating": 4.7, "icon": "🌸",
        "profile": {"energy": 3, "nature": 2, "romance": 4, "adventurous": 3, "classic": 4, "warmth": 4, "bold": 3},
        "mood_fit": ["Romantic & Dreamy", "Energized & Upbeat"],
        "season_fit": ["Spring – Fresh & Hopeful", "Summer – Warm & Vibrant"],
        "occasion_fit": ["Special occasions", "Daily wear"],
    },
    {
        "id": 11, "name": "Bleu de Chanel", "brand": "Chanel",
        "notes": "Lemon, Bergamot, Grapefruit, Ginger, Incense, Sandalwood, Cedar",
        "accords": ["Woody", "Fresh", "Aromatic"], "rating": 4.8, "icon": "🔵",
        "profile": {"energy": 3, "nature": 3, "romance": 2, "adventurous": 4, "classic": 4, "warmth": 3, "bold": 4},
        "mood_fit": ["Energized & Upbeat", "Calm & Reflective"],
        "season_fit": ["Winter – Cool & Refined", "Autumn – Rich & Complex"],
        "occasion_fit": ["Work & professional", "Daily wear"],
    },
]

MOODS    = ["Energized & Upbeat", "Calm & Reflective", "Romantic & Dreamy", "Mysterious & Intense"]
SEASONS  = ["Spring – Fresh & Hopeful", "Summer – Warm & Vibrant", "Autumn – Rich & Complex", "Winter – Cool & Refined"]
OCCASIONS= ["Daily wear", "Special occasions", "Work & professional", "Evening & nightlife"]


def score_perfume(perfume, sample):
    """Compute similarity score between a user sample and a perfume's ideal profile."""
    profile = perfume["profile"]
    score = 0

    # Weighted euclidean distance on numeric traits
    weights = {"energy":1.2, "nature":1.0, "romance":1.5, "adventurous":1.0,
               "classic":1.0, "warmth":0.8, "bold":1.3}
    dist = sum(w * (sample[k] - profile[k])**2 for k, w in weights.items())
    score += max(0, 10 - dist)  # higher = better fit

    # Bonus for mood/season/occasion match
    if sample["mood"] in perfume["mood_fit"]:       score += 4
    if sample["season"] in perfume["season_fit"]:   score += 3
    if sample["occasion"] in perfume["occasion_fit"]:score += 3

    # Slight rating bonus
    score += (perfume["rating"] - 4.0) * 2

    return score


def generate_sample():
    """Generate one random personality sample and find its best perfume label."""
    sample = {
        "energy":      random.randint(1, 5),
        "nature":      random.randint(1, 5),
        "romance":     random.randint(1, 5),
        "adventurous": random.randint(1, 5),
        "classic":     random.randint(1, 5),
        "warmth":      random.randint(1, 5),
        "bold":        random.randint(1, 5),
        "mood":        random.choice(MOODS),
        "season":      random.choice(SEASONS),
        "occasion":    random.choice(OCCASIONS),
    }

    # Score all perfumes
    scores = [(p["id"], score_perfume(p, sample)) for p in PERFUMES]
    scores.sort(key=lambda x: -x[1])

    # Add slight noise for diversity
    sample["label"] = scores[0][0]

    # Encode categoricals as integers
    sample["mood_enc"]    = MOODS.index(sample["mood"])
    sample["season_enc"]  = SEASONS.index(sample["season"])
    sample["occasion_enc"]= OCCASIONS.index(sample["occasion"])

    return sample


def generate_dataset(n=5000):
    rows = [generate_sample() for _ in range(n)]
    df = pd.DataFrame(rows)
    feature_cols = ["energy","nature","romance","adventurous","classic","warmth","bold",
                    "mood_enc","season_enc","occasion_enc"]
    return df[feature_cols + ["label"]]


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("perfume_data.csv", index=False)
    print(f"✓ Generated {len(df)} training samples")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nSample row:")
    print(df.head(2).to_string())
