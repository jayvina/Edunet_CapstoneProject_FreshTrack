"""
freshtrack_ml.py
=================
FreshTrack — Full AI/ML Edition (Edunet Foundation Project)

Ties together three trained ML models into one working app:

  * RandomForestRegressor   -> predicts how many days an item will last
  * RandomForestClassifier  -> predicts the probability it gets wasted
  * KMeans clustering       -> groups items into High/Medium/Low
                                waste-risk clusters

Run order:
    1. python generate_data.py
    2. python train_models.py
    3. python freshtrack_ml.py   <- this file

If models/ doesn't exist yet, this file will tell you to run the first
two scripts first.
"""

import json
import os
import re
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"
DATA_FILE = "freshtrack_inventory.json"

CATEGORY_LIST = [
    "leafy greens", "berries", "banana", "apple", "tomato", "citrus",
    "milk", "yogurt", "cheese", "eggs", "bread", "chicken (raw)",
    "fish (raw)", "cooked leftovers", "rice (cooked)", "root vegetables",
    "packaged snacks",
]
STORAGE_LIST = ["fridge", "freezer", "pantry"]

# category -> (moisture_level, perishability_index) — same profile used to
# generate training data, so a typed item can be turned into a feature row.
CATEGORY_PROFILE = {
    "leafy greens":       (0.85, 9),
    "berries":            (0.80, 9),
    "banana":             (0.55, 8),
    "apple":              (0.45, 4),
    "tomato":             (0.70, 7),
    "citrus":             (0.40, 4),
    "milk":               (0.90, 8),
    "yogurt":             (0.75, 6),
    "cheese":             (0.35, 3),
    "eggs":               (0.20, 2),
    "bread":              (0.30, 7),
    "chicken (raw)":      (0.65, 10),
    "fish (raw)":         (0.70, 10),
    "cooked leftovers":   (0.60, 9),
    "rice (cooked)":      (0.55, 7),
    "root vegetables":    (0.25, 3),
    "packaged snacks":    (0.05, 1),
}
STORAGE_TEMP = {"fridge": 4.0, "freezer": -18.0, "pantry": 23.0}

KNOWN_ITEM_CATEGORIES = {
    "spinach": "leafy greens", "lettuce": "leafy greens", "kale": "leafy greens",
    "strawberr": "berries", "blueberr": "berries", "raspberr": "berries",
    "banana": "banana", "apple": "apple", "tomato": "tomato",
    "orange": "citrus", "lemon": "citrus", "lime": "citrus",
    "milk": "milk", "yogurt": "yogurt", "curd": "yogurt",
    "cheese": "cheese", "paneer": "cheese", "egg": "eggs", "bread": "bread",
    "chicken": "chicken (raw)", "fish": "fish (raw)", "prawn": "fish (raw)",
    "leftover": "cooked leftovers", "rice": "rice (cooked)",
    "potato": "root vegetables", "onion": "root vegetables", "carrot": "root vegetables",
    "chips": "packaged snacks", "biscuit": "packaged snacks",
}

RECIPES = {
    "leafy greens": ["Sauteed greens with garlic", "Spinach dal", "Green smoothie"],
    "berries": ["Berry smoothie", "Berry oatmeal topping", "Quick berry compote"],
    "banana": ["Banana pancakes", "Banana bread", "Banana smoothie"],
    "apple": ["Apple cinnamon oatmeal", "Baked apples", "Apple salad"],
    "tomato": ["Tomato soup", "Pasta sauce", "Tomato chutney"],
    "citrus": ["Citrus salad dressing", "Fresh juice", "Citrus marinade"],
    "milk": ["Pancakes", "White sauce pasta", "Rice pudding"],
    "yogurt": ["Yogurt smoothie", "Raita", "Marinade for curry"],
    "cheese": ["Grilled cheese sandwich", "Cheesy omelette", "Quesadilla"],
    "eggs": ["Omelette", "Fried rice with egg", "Egg curry"],
    "bread": ["French toast", "Bread pudding", "Croutons"],
    "chicken (raw)": ["Stir-fried chicken", "Chicken curry", "Grilled chicken"],
    "fish (raw)": ["Pan-seared fish", "Fish curry", "Fish tacos"],
    "cooked leftovers": ["Fried rice remix", "Stuffed wrap", "Leftover soup"],
    "rice (cooked)": ["Fried rice", "Rice pudding", "Rice cutlets"],
    "root vegetables": ["Vegetable stir-fry", "Roasted root veggies", "Vegetable curry"],
    "packaged snacks": ["Use as toppings before they go stale"],
}


def classify_item(item_name: str) -> str:
    """Simulated photo-recognition step: item name -> food category."""
    name = item_name.lower()
    for keyword, category in KNOWN_ITEM_CATEGORIES.items():
        if keyword in name:
            return category
    return "packaged snacks"  # safe fallback category


def suggest_storage(category: str, raw_name: str) -> str:
    """Auto-suggest storage type based on food category and keyword cues (e.g. 'frozen')."""
    name_lower = raw_name.lower()
    if "freezer" in name_lower or "frozen" in name_lower:
        return "freezer"

    fridge_categories = {
        "leafy greens", "berries", "milk", "yogurt", "cheese", "eggs",
        "chicken (raw)", "fish (raw)", "cooked leftovers", "rice (cooked)"
    }
    pantry_categories = {
        "banana", "apple", "tomato", "citrus", "root vegetables", "packaged snacks", "bread"
    }

    if category in fridge_categories:
        return "fridge"
    elif category in pantry_categories:
        return "pantry"

    return "fridge"  # default fallback


class FreshTrackML:
    def __init__(self):
        if not os.path.exists(MODEL_DIR):
            raise SystemExit(
                "No trained models found. Run these first:\n"
                "  python generate_data.py\n"
                "  python train_models.py\n"
            )
        self.regressor = joblib.load(f"{MODEL_DIR}/shelf_life_regressor.joblib")
        self.classifier = joblib.load(f"{MODEL_DIR}/waste_risk_classifier.joblib")
        cluster_bundle = joblib.load(f"{MODEL_DIR}/cluster_model.joblib")
        self.kmeans = cluster_bundle["kmeans"]
        self.cluster_scaler = cluster_bundle["scaler"]
        self.cluster_name_map = cluster_bundle["cluster_name_map"]
        with open(f"{MODEL_DIR}/metrics.json") as f:
            self.metrics = json.load(f)

        self.items = []
        self.waste_log = []
        self._load()

    # ---------------- persistence ----------------
    def _load(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.waste_log = data.get("waste_log", [])

    def _save(self):
        with open(DATA_FILE, "w") as f:
            json.dump({"items": self.items, "waste_log": self.waste_log}, f, indent=2)

    # ---------------- feature building ----------------
    def _household_usage_speed(self) -> float:
        """Learn this household's usage speed from its own waste log —
        this is FreshTrack's 'waste log improves predictions over time'
        feature in action. Starts neutral (0.6) until enough data exists.
        """
        if len(self.waste_log) < 3:
            return 0.6
        recent = self.waste_log[-15:]
        wasted_ratio = sum(1 for w in recent if w["wasted"]) / len(recent)
        return float(np.clip(1 - wasted_ratio, 0.1, 0.95))

    def _build_feature_row(self, category, storage, quantity):
        moisture, perishability = CATEGORY_PROFILE.get(category, (0.5, 5))
        temp = STORAGE_TEMP[storage]
        usage_speed = self._household_usage_speed()
        return pd.DataFrame([{
            "category": category,
            "storage_type": storage,
            "quantity": quantity,
            "purchase_temperature_c": temp,
            "moisture_level": moisture,
            "perishability_index": perishability,
            "household_usage_speed": usage_speed,
        }])

    # ---------------- ML-backed actions ----------------
    def add_item(self, raw_name: str, storage: str, quantity: int = 1):
        category = classify_item(raw_name)
        row = self._build_feature_row(category, storage, quantity)

        predicted_days = max(1, round(float(self.regressor.predict(row)[0])))
        waste_probability = float(self.classifier.predict_proba(row)[0][1])

        cluster_features = row[["moisture_level", "perishability_index"]].copy()
        cluster_features["shelf_life_days"] = predicted_days
        cluster_features = cluster_features[["shelf_life_days", "moisture_level", "perishability_index"]]
        scaled = self.cluster_scaler.transform(cluster_features)
        cluster_id = int(self.kmeans.predict(scaled)[0])
        risk_group = self.cluster_name_map[cluster_id]

        purchased = datetime.now()
        expiry = purchased + timedelta(days=predicted_days)

        item = {
            "name": raw_name.strip(),
            "category": category,
            "storage_type": storage,
            "quantity": quantity,
            "purchased_on": purchased.strftime("%Y-%m-%d"),
            "expiry_estimate": expiry.strftime("%Y-%m-%d"),
            "predicted_shelf_life_days": predicted_days,
            "waste_risk_probability": round(waste_probability, 3),
            "risk_cluster": risk_group,
        }
        self.items.append(item)
        self._save()

        print(
            f"\nAdded '{item['name']}' -> category: {category} | storage: {storage}\n"
            f"  Predicted shelf life : {predicted_days} days (use by {item['expiry_estimate']})\n"
            f"  Waste risk (ML)      : {round(waste_probability * 100, 1)}%\n"
            f"  Risk cluster (KMeans): {risk_group}"
        )

    def use_it_soon(self, days_threshold: int = 3):
        today = datetime.now().date()
        soon = []
        for item in self.items:
            expiry = datetime.strptime(item["expiry_estimate"], "%Y-%m-%d").date()
            days_left = (expiry - today).days
            if days_left <= days_threshold:
                soon.append((item, days_left))
        # Prioritize by days left first, then by ML waste-risk probability
        soon.sort(key=lambda pair: (pair[1], -pair[0]["waste_risk_probability"]))
        return soon

    def mark_used(self, name: str):
        for item in list(self.items):
            if item["name"].lower() == name.lower():
                self.items.remove(item)
                self.waste_log.append({**item, "wasted": 0, "logged_on": datetime.now().strftime("%Y-%m-%d")})
                self._save()
                print(f"'{name}' marked as used. One less item wasted!")
                return
        print(f"No active item named '{name}' found.")

    def log_waste(self, name: str):
        for item in list(self.items):
            if item["name"].lower() == name.lower():
                self.items.remove(item)
                self.waste_log.append({**item, "wasted": 1, "logged_on": datetime.now().strftime("%Y-%m-%d")})
                self._save()
                print(
                    f"Logged waste for '{name}'. This household's usage-speed profile "
                    f"will adjust future waste-risk predictions."
                )
                return
        print(f"No active item named '{name}' found.")

    def waste_summary(self):
        if not self.waste_log:
            print("No history yet.")
            return
        wasted = [w for w in self.waste_log if w["wasted"] == 1]
        used = [w for w in self.waste_log if w["wasted"] == 0]
        print(f"\nUsed in time: {len(used)}   |   Wasted: {len(wasted)}")
        if wasted:
            by_category = {}
            for w in wasted:
                by_category.setdefault(w["category"], 0)
                by_category[w["category"]] += 1
            print("Wasted items by category:")
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                print(f"  - {cat}: {count}")
        print(f"Current household usage-speed score: {self._household_usage_speed():.2f} (1.0 = wastes almost nothing)")

    def show_model_report(self):
        m = self.metrics
        print("\n===== ML Model Report =====")
        print("\nRegression (shelf-life prediction):")
        print(f"  MAE: {m['regression']['mae_days']} days")
        print(f"  R^2: {m['regression']['r2_score']}")
        print("\nClassification (waste risk prediction):")
        for k, v in m["classification"].items():
            print(f"  {k}: {v}")
        print("\nClustering (KMeans waste-risk groups):")
        print(f"  Silhouette score: {m['clustering']['silhouette_score']}")
        for cluster_id, stats in m["clustering"]["clusters"].items():
            print(
                f"  Cluster {cluster_id} [{stats['risk_label']}]: "
                f"avg shelf life {stats['avg_shelf_life']}d, "
                f"waste rate {stats['waste_rate']*100:.0f}%, "
                f"n={int(stats['count'])}"
            )
        print("\nSee plots/clusters.png, plots/confusion_matrix.png, "
              "plots/feature_importance.png for visuals.")

    def import_bill(self, file_path: str):
        """Parse a grocery bill text file, run ML predictions, and import items to inventory."""
        if not os.path.exists(file_path):
            print(f"Error: Bill file '{file_path}' not found.")
            return

        print(f"\nParsing bill '{file_path}'...")
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("===") or line.startswith("---") or line.startswith("***"):
                    continue

                # Parse quantity and item name
                qty = 1
                item_name = line
                match = re.match(r"^(\d+)\s*[xX]?\s+(.+)$", line)
                if match:
                    qty = int(match.group(1))
                    item_name = match.group(2)

                # Remove trailing prices/currency indicators (e.g. $1.99 or 3.50)
                item_name = re.sub(r"\s*[\$]?\d+[\.,]\d{2}\s*$", "", item_name).strip()

                if not item_name:
                    continue

                category = classify_item(item_name)
                storage = suggest_storage(category, item_name)

                # Predict shelf life, waste risk, and cluster using ML models
                row = self._build_feature_row(category, storage, qty)
                predicted_days = max(1, round(float(self.regressor.predict(row)[0])))
                waste_probability = float(self.classifier.predict_proba(row)[0][1])

                cluster_features = row[["moisture_level", "perishability_index"]].copy()
                cluster_features["shelf_life_days"] = predicted_days
                cluster_features = cluster_features[["shelf_life_days", "moisture_level", "perishability_index"]]
                scaled = self.cluster_scaler.transform(cluster_features)
                cluster_id = int(self.kmeans.predict(scaled)[0])
                risk_group = self.cluster_name_map[cluster_id]

                purchased = datetime.now()
                expiry = purchased + timedelta(days=predicted_days)

                item = {
                    "name": item_name,
                    "category": category,
                    "storage_type": storage,
                    "quantity": qty,
                    "purchased_on": purchased.strftime("%Y-%m-%d"),
                    "expiry_estimate": expiry.strftime("%Y-%m-%d"),
                    "predicted_shelf_life_days": predicted_days,
                    "waste_risk_probability": round(waste_probability, 3),
                    "risk_cluster": risk_group,
                }
                self.items.append(item)
                count += 1

                print(
                    f"  - Imported: {qty}x '{item_name}' | category: {category} | "
                    f"storage: {storage} | shelf life: {predicted_days} days | "
                    f"risk: {round(waste_probability * 100, 1)}% ({risk_group})"
                )

        if count > 0:
            self._save()
            print(f"\nSuccessfully imported and saved {count} item(s) from grocery bill!")
        else:
            print("\nNo items could be parsed from the bill file.")


MENU = """
================= FreshTrack — AI/ML Edition =================
 1. Add grocery item (ML predicts shelf life + waste risk + cluster)
 2. View 'Use-It-Soon' reminder list (ML-prioritized)
 3. Get recipe suggestions for a category
 4. Mark an item as used
 5. Log an item as wasted (feeds back into ML predictions)
 6. View waste summary
 7. View full inventory
 8. View ML model report (accuracy, R^2, clusters)
 9. Upload/Parse grocery bill (auto-import all items)
 10. Exit
================================================================
"""


def choose_from_list(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def run():
    app = FreshTrackML()

    while True:
        print(MENU)
        choice = input("Choose an option (1-10): ").strip()

        if choice == "1":
            name = input("Item name (e.g. 'spinach', 'milk carton'): ").strip()
            if not name:
                continue
            storage = choose_from_list("Storage location:", STORAGE_LIST)
            qty = input("Quantity [default 1]: ").strip()
            qty = int(qty) if qty.isdigit() else 1
            app.add_item(name, storage, qty)

        elif choice == "2":
            threshold = input("Show items expiring within how many days? [default 3]: ").strip()
            threshold = int(threshold) if threshold.isdigit() else 3
            soon = app.use_it_soon(threshold)
            if not soon:
                print("Nothing expiring soon.")
            else:
                print(f"\nUse within {threshold} days (highest waste-risk first):")
                for item, days_left in soon:
                    status = "EXPIRED" if days_left < 0 else f"{days_left} day(s) left"
                    print(
                        f"  - {item['name']} ({item['category']}) — {status} | "
                        f"waste risk: {item['waste_risk_probability']*100:.0f}% | "
                        f"{item['risk_cluster']}"
                    )

        elif choice == "3":
            category = choose_from_list("Pick a category:", CATEGORY_LIST)
            print(f"\nRecipe ideas for '{category}':")
            for r in RECIPES.get(category, ["Search online for recipes"]):
                print(f"  - {r}")

        elif choice == "4":
            name = input("Item name to mark as used: ").strip()
            app.mark_used(name)

        elif choice == "5":
            name = input("Item name to log as wasted: ").strip()
            app.log_waste(name)

        elif choice == "6":
            app.waste_summary()

        elif choice == "7":
            if not app.items:
                print("Inventory is empty.")
            else:
                print("\nCurrent inventory:")
                for item in app.items:
                    print(
                        f"  - {item['name']} ({item['category']}, {item['storage_type']}) | "
                        f"use by {item['expiry_estimate']} | "
                        f"waste risk {item['waste_risk_probability']*100:.0f}% | "
                        f"{item['risk_cluster']}"
                    )

        elif choice == "8":
            app.show_model_report()

        elif choice == "9":
            path = input("Enter path to grocery bill text file: ").strip()
            if path:
                app.import_bill(path)

        elif choice == "10":
            print("Goodbye! Keep FreshTrack running to cut down food waste.")
            break

        else:
            print("Invalid choice, please pick 1-10.")


if __name__ == "__main__":
    run()
