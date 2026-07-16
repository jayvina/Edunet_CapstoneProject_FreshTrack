import json
import os
import re
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Constants and paths
MODEL_DIR = "models"
DATA_FILE = "freshtrack_inventory.json"

CATEGORY_LIST = [
    "leafy greens", "berries", "banana", "apple", "tomato", "citrus",
    "milk", "yogurt", "cheese", "eggs", "bread", "chicken (raw)",
    "fish (raw)", "cooked leftovers", "rice (cooked)", "root vegetables",
    "packaged snacks",
]
STORAGE_LIST = ["fridge", "freezer", "pantry"]

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

# Load Models
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(
        "No trained models found. Train models first using `python train_models.py`"
    )

regressor = joblib.load(f"{MODEL_DIR}/shelf_life_regressor.joblib")
classifier = joblib.load(f"{MODEL_DIR}/waste_risk_classifier.joblib")
cluster_bundle = joblib.load(f"{MODEL_DIR}/cluster_model.joblib")
kmeans = cluster_bundle["kmeans"]
cluster_scaler = cluster_bundle["scaler"]
cluster_name_map = cluster_bundle["cluster_name_map"]

with open(f"{MODEL_DIR}/metrics.json") as f:
    model_metrics = json.load(f)


# Helper Functions
def classify_item(item_name: str) -> str:
    name = item_name.lower()
    for keyword, category in KNOWN_ITEM_CATEGORIES.items():
        if keyword in name:
            return category
    return "packaged snacks"

def suggest_storage(category: str, raw_name: str) -> str:
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
    return "fridge"


class FreshTrackState:
    def __init__(self):
        self.items = []
        self.waste_log = []
        self.load()

    def load(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.waste_log = data.get("waste_log", [])

    def save(self):
        with open(DATA_FILE, "w") as f:
            json.dump({"items": self.items, "waste_log": self.waste_log}, f, indent=2)

    def household_usage_speed(self) -> float:
        if len(self.waste_log) < 3:
            return 0.6
        recent = self.waste_log[-15:]
        wasted_ratio = sum(1 for w in recent if w["wasted"]) / len(recent)
        return float(np.clip(1 - wasted_ratio, 0.1, 0.95))

    def build_feature_row(self, category, storage, quantity):
        moisture, perishability = CATEGORY_PROFILE.get(category, (0.5, 5))
        temp = STORAGE_TEMP[storage]
        usage_speed = self.household_usage_speed()
        return pd.DataFrame([{
            "category": category,
            "storage_type": storage,
            "quantity": quantity,
            "purchase_temperature_c": temp,
            "moisture_level": moisture,
            "perishability_index": perishability,
            "household_usage_speed": usage_speed,
        }])

    def process_and_add_item(self, raw_name, storage, quantity):
        category = classify_item(raw_name)
        row = self.build_feature_row(category, storage, quantity)

        predicted_days = max(1, round(float(regressor.predict(row)[0])))
        waste_probability = float(classifier.predict_proba(row)[0][1])

        cluster_features = row[["moisture_level", "perishability_index"]].copy()
        cluster_features["shelf_life_days"] = predicted_days
        cluster_features = cluster_features[["shelf_life_days", "moisture_level", "perishability_index"]]
        scaled = cluster_scaler.transform(cluster_features)
        cluster_id = int(kmeans.predict(scaled)[0])
        risk_group = cluster_name_map[cluster_id]

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
        return item


state = FreshTrackState()

# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    state.load()
    # Calculate some helper aggregates
    wasted_count = sum(1 for w in state.waste_log if w["wasted"] == 1)
    used_count = sum(1 for w in state.waste_log if w["wasted"] == 0)
    total_logged = len(state.waste_log)
    waste_rate = round((wasted_count / total_logged * 100), 1) if total_logged > 0 else 0.0

    return jsonify({
        "items": state.items,
        "waste_log": state.waste_log,
        "household_usage_speed": round(state.household_usage_speed(), 2),
        "aggregates": {
            "total_active": len(state.items),
            "used": used_count,
            "wasted": wasted_count,
            "waste_rate": waste_rate
        },
        "model_metrics": model_metrics,
        "recipes": RECIPES
    })


@app.route("/api/inventory/add", methods=["POST"])
def add_inventory_item():
    data = request.json or {}
    name = data.get("name", "").strip()
    storage = data.get("storage", "fridge").strip()
    qty = data.get("quantity", 1)
    
    if not name:
        return jsonify({"error": "Item name is required"}), 400
    if storage not in STORAGE_LIST:
        return jsonify({"error": "Invalid storage location"}), 400

    item = state.process_and_add_item(name, storage, qty)
    state.save()
    return jsonify({"success": True, "item": item})


@app.route("/api/inventory/action", methods=["POST"])
def inventory_action():
    data = request.json or {}
    name = data.get("name", "").strip()
    action = data.get("action", "").strip() # "used" or "wasted"

    if not name or action not in ["used", "wasted"]:
        return jsonify({"error": "Invalid arguments"}), 400

    for item in list(state.items):
        if item["name"].lower() == name.lower():
            state.items.remove(item)
            wasted_bit = 1 if action == "wasted" else 0
            state.waste_log.append({
                **item, 
                "wasted": wasted_bit, 
                "logged_on": datetime.now().strftime("%Y-%m-%d")
            })
            state.save()
            return jsonify({"success": True, "message": f"Item marked as {action}."})

    return jsonify({"error": "Item not found"}), 404


@app.route("/api/upload", methods=["POST"])
def upload_receipt():
    # Verify file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read items. If name contains sample_receipt or we want to simulate the receipt,
    # we return a rich selection of items.
    filename = file.filename.lower()
    
    # We will simulate the OCR process
    # If the user uploads our specific sample receipt or any image, we parse items
    if "sample" in filename or "receipt" in filename:
        mock_items = [
            ("organic spinach", 2),
            ("fresh whole milk", 1),
            ("bananas", 4),
            ("raw chicken breasts", 1),
            ("frozen fish fillets", 3),
            ("whole wheat bread", 2),
            ("sweet apples", 6),
            ("potato", 3)
        ]
    else:
        # Generate some randomized realistic groceries for other uploads
        mock_items = [
            ("fresh strawberries", 1),
            ("curd yogurt", 2),
            ("packaged potato chips", 2),
            ("cheddar cheese block", 1),
            ("raw chicken wings", 1)
        ]

    imported_items = []
    for raw_name, qty in mock_items:
        item = state.process_and_add_item(raw_name, suggest_storage(classify_item(raw_name), raw_name), qty)
        imported_items.append(item)
        
    state.save()
    return jsonify({
        "success": True, 
        "imported_count": len(imported_items),
        "items": imported_items
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
