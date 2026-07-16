"""
generate_data.py
=================
Builds a synthetic household grocery dataset for FreshTrack's ML models.

Real Teachable-Machine-style photo data isn't available for a student
project, but the *downstream* ML pipeline (regression, classification,
clustering) needs real numeric/categorical data to train on. This script
generates a realistic synthetic dataset that mirrors how such data would
look if collected from FreshTrack's waste log over time:

    category, storage_type, quantity, purchase_temperature_c,
    moisture_level, perishability_index  -> features
    shelf_life_days                      -> regression target
    wasted (0/1)                         -> classification target

Run this once before train_models.py.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 3000

# category -> (base_shelf_life_days_in_fridge, moisture_level 0-1, perishability_index 1-10)
CATEGORY_PROFILE = {
    "leafy greens":       (5,  0.85, 9),
    "berries":            (5,  0.80, 9),
    "banana":             (5,  0.55, 8),
    "apple":              (21, 0.45, 4),
    "tomato":             (7,  0.70, 7),
    "citrus":             (14, 0.40, 4),
    "milk":               (7,  0.90, 8),
    "yogurt":             (10, 0.75, 6),
    "cheese":             (21, 0.35, 3),
    "eggs":               (28, 0.20, 2),
    "bread":              (5,  0.30, 7),
    "chicken (raw)":      (2,  0.65, 10),
    "fish (raw)":         (2,  0.70, 10),
    "cooked leftovers":   (3,  0.60, 9),
    "rice (cooked)":      (4,  0.55, 7),
    "root vegetables":    (21, 0.25, 3),
    "packaged snacks":    (90, 0.05, 1),
}

STORAGE_TYPES = {
    # storage -> (shelf_life_multiplier, typical_temperature_c)
    "fridge":  (1.0, 4.0),
    "freezer": (3.5, -18.0),
    "pantry":  (0.55, 23.0),
}

categories = list(CATEGORY_PROFILE.keys())
storages = list(STORAGE_TYPES.keys())

rows = []
for _ in range(N_SAMPLES):
    category = RNG.choice(categories)
    base_life, moisture_base, perish_base = CATEGORY_PROFILE[category]

    # Non-perishables/packaged snacks are rarely frozen; raw meat/fish/dairy
    # are rarely left in the pantry. Bias storage choice realistically.
    if category == "packaged snacks":
        storage = RNG.choice(storages, p=[0.05, 0.05, 0.90])
    elif category in ("chicken (raw)", "fish (raw)", "milk", "yogurt", "cheese"):
        storage = RNG.choice(storages, p=[0.65, 0.30, 0.05])
    else:
        storage = RNG.choice(storages, p=[0.55, 0.15, 0.30])

    multiplier, temp_base = STORAGE_TYPES[storage]
    temperature = temp_base + RNG.normal(0, 1.5)

    moisture = float(np.clip(moisture_base + RNG.normal(0, 0.05), 0, 1))
    perishability = float(np.clip(perish_base + RNG.normal(0, 0.5), 1, 10))
    quantity = max(1, int(RNG.poisson(3)))

    shelf_life_days = max(1, round(base_life * multiplier + RNG.normal(0, base_life * 0.12)))

    # Simulate how quickly this household actually gets around to using the
    # item. Faster-moving households waste less; slower ones waste more.
    # `household_usage_speed` is what FreshTrack's waste log would learn
    # per household over time (0 = very slow to use food, 1 = very fast) —
    # a real, observable feature, not a leak of the label itself.
    household_speed = RNG.choice(["fast", "medium", "slow"], p=[0.3, 0.45, 0.25])
    speed_factor = {"fast": 0.5, "medium": 0.75, "slow": 1.05}[household_speed]
    usage_speed_score = {"fast": 0.85, "medium": 0.55, "slow": 0.25}[household_speed]
    usage_speed_score = float(np.clip(usage_speed_score + RNG.normal(0, 0.07), 0, 1))

    usage_delay_days = max(0.5, RNG.normal(shelf_life_days * speed_factor, shelf_life_days * 0.2))
    wasted = int(usage_delay_days > shelf_life_days)

    rows.append({
        "category": category,
        "storage_type": storage,
        "quantity": quantity,
        "purchase_temperature_c": round(temperature, 1),
        "moisture_level": round(moisture, 3),
        "perishability_index": round(perishability, 2),
        "household_usage_speed": round(usage_speed_score, 3),
        "shelf_life_days": shelf_life_days,
        "wasted": wasted,
    })

df = pd.DataFrame(rows)
df.to_csv("freshtrack_dataset.csv", index=False)

print(f"Generated {len(df)} rows -> freshtrack_dataset.csv")
print(df.head())
print("\nWaste rate:", round(df["wasted"].mean() * 100, 1), "%")
