# FreshTrack 🥦📸

**AI-powered kitchen assistant that turns food-waste prevention into a habit — not a guessing game.**

FreshTrack helps households track groceries, predicts when items will spoil, and nudges users to use food before it's wasted — all from a single photo, with zero extra hardware.

---

## 🚩 The Problem

Nearly a third of all food produced globally is wasted, and a huge chunk of that happens right at home — groceries get forgotten, expiry dates slip by, and leftovers get binned simply because nobody was tracking what needed to be used. Every wasted item also wastes the water, energy, and emissions that went into producing it.

## The Solution

Users photograph their groceries or a receipt after a shopping trip. FreshTrack identifies the items, estimates how long each will stay fresh, and builds a running **"use-it-soon" list** — with reminders before things spoil and recipe suggestions built around what's about to expire.

## Key Features

- 📸 **Zero manual entry** — one photo of groceries or a receipt logs the whole haul via OCR + image recognition
- 🔮 **Predictive, not reactive** — shelf-life estimates warn users *before* food spoils, not after
- 🍽 **Turns waste into meals** — expiring items automatically surface as recipe suggestions
- 📈 **Learns over time** — a waste log lets users mark what was actually thrown out, sharpening future predictions

## How It Works

```
📷 Capture  →  🔎 Recognize  →  📅 Predict  →  🔔 Remind  →  🍳 Suggest
```

1. **Capture** — photo of groceries or a receipt
2. **Recognize** — a CNN classifies items / OCR reads receipt text
3. **Predict** — shelf-life lookup per food category
4. **Remind** — use-it-soon list + daily/weekly alerts
5. **Suggest** — recipes built around expiring items

🔁 **Feedback loop:** users log what was actually thrown away in the Waste Log, and the shelf-life model recalibrates over time.

## Who It's For

- **Households** — cutting grocery bills and waste week to week
- **Hostels & mess canteens** — tracking bulk perishables for large groups
- **Small restaurants & cafés** — managing perishable inventory without expensive POS systems
- **Sustainability communities** — apartments and eco-groups running waste-reduction drives

## Tech Stack

**Software & AI**
- Python — core language
- Teachable Machine / CNN (TensorFlow-Keras) — classifies food items from photos
- Tesseract OCR — reads receipt text
- Rule-based shelf-life lookup — beginner-safe, upgradable to a full ML model
- Streamlit — dashboard for reminders, waste log, and recipes

**Hardware**
- Just a smartphone or laptop camera — no sensors or microcontrollers required

**Data**
- Self-collected grocery photo dataset
- USDA FoodKeeper shelf-life reference data

## Future Perspective

- 📡 **Smart fridge integration** — auto inventory updates via smart-fridge sensors
- 🤝 **Community food-sharing** — let neighbors claim surplus food nearing expiry
- 🧾 **Retailer partnerships** — pull structured expiry data from e-receipts and loyalty apps
- 🌎 **Regional shelf-life models** — localize predictions for climate, diet, and produce variety

## Conclusion

Household food waste is a solvable, everyday problem — not just an industrial-scale one. FreshTrack lowers the effort of tracking groceries to a single photo, making waste prevention a habit rather than a chore. A lightweight AI stack (image recognition + OCR + a shelf-life lookup) is enough to deliver real value without heavy infrastructure.

## Team

- **Team Lead:** Vasukumar Chauhan
- **Team Member:** Jayvina Dhedhi
- **Mentor:** Sagar Chouhan
- **College:** V.V.P. Engineering College, Rajkot

---
