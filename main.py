import io
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI()

print("Loading InceptionV3 model...")
model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=True)
print("Model loaded.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((299, 299))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def query_open_food_facts(food_name: str) -> dict:
    clean = food_name.replace("_", " ")
    try:
        r = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={
                "search_terms": clean,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 1,
                "fields": "nutriments",
            },
            timeout=5,
        )
        products = r.json().get("products", [])
        if products:
            n = products[0].get("nutriments", {})
            return {
                "calories": round(float(n.get("energy-kcal_100g", 0))),
                "protein_g": round(float(n.get("proteins_100g", 0)), 1),
                "carbs_g": round(float(n.get("carbohydrates_100g", 0)), 1),
                "fat_g": round(float(n.get("fat_100g", 0)), 1),
            }
    except Exception as e:
        print(f"OpenFoodFacts error: {e}")
    return _fallback_nutrition(food_name)

def _fallback_nutrition(food_name: str) -> dict:
    high_cal = ["beef", "pork", "steak", "ribs", "burger", "pizza", "cake", "donut", "fries"]
    low_cal = ["salad", "sashimi", "edamame", "miso", "seaweed", "ceviche"]
    name = food_name.lower()
    if any(h in name for h in high_cal):
        return {"calories": 450, "protein_g": 25.0, "carbs_g": 35.0, "fat_g": 22.0}
    if any(l in name for l in low_cal):
        return {"calories": 120, "protein_g": 8.0, "carbs_g": 10.0, "fat_g": 3.0}
    return {"calories": 280, "protein_g": 15.0, "carbs_g": 30.0, "fat_g": 10.0}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)
        preds = model.predict(processed)
        decoded = tf.keras.applications.inception_v3.decode_predictions(preds, top=1)[0]
        top_label = decoded[0][1]
        confidence = float(decoded[0][2])
        food_name = top_label.replace("_", " ").lower()
        nutrition = query_open_food_facts(top_label)
        return JSONResponse({
            "items": [{
                "food_name": food_name,
                "calories": nutrition["calories"],
                "protein_g": nutrition["protein_g"],
                "carbs_g": nutrition["carbs_g"],
                "fat_g": nutrition["fat_g"],
                "confidence": "high" if confidence > 0.6 else "medium"
            }]
        })
    except Exception as e:
        print(f"Analyze error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
