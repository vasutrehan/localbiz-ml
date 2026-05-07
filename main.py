"""
LocalBiz ML Recommendation Microservice
FastAPI + Scikit-learn
Collaborative filtering + content-based + location-aware ranking
"""

import os
import math
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = FastAPI(title="LocalBiz ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB connection ──
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/localbiz"))
db = client["localbiz"]

# ── Haversine distance (km) ──
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Load data from MongoDB ──
def load_businesses():
    businesses = list(db.businesses.find({"isActive": True}, {
        "_id": 1, "name": 1, "category": 1, "rating": 1,
        "totalReviews": 1, "priceRange": 1, "isVerified": 1,
        "location": 1, "tags": 1
    }))
    return businesses


def load_user_interactions(user_id: str):
    """
    Gather: saved businesses, reviewed businesses, viewed businesses.
    Returns list of {businessId, weight} dicts.
    """
    interactions = []

    # Saved businesses → weight 3
    user = db.users.find_one({"_id": user_id}, {"savedBusinessIds": 1})
    if user:
        for bid in user.get("savedBusinessIds", []):
            interactions.append({"businessId": str(bid), "weight": 3.0})

    # Reviews → weight = rating / 5 * 3
    reviews = db.reviews.find({"user": user_id}, {"business": 1, "rating": 1})
    for r in reviews:
        interactions.append({"businessId": str(r["business"]), "weight": r["rating"] / 5 * 3})

    return interactions


# ── Content-based feature vector for a business ──
CATEGORIES = ["food", "health", "shopping", "services", "education", "sports", "other"]

def business_feature_vector(biz):
    cat_vec = [1 if biz.get("category") == c else 0 for c in CATEGORIES]
    rating_norm = (biz.get("rating", 0) or 0) / 5.0
    reviews_norm = min((biz.get("totalReviews", 0) or 0) / 500.0, 1.0)
    price_norm = (biz.get("priceRange", 2) or 2) / 4.0
    verified = 1.0 if biz.get("isVerified") else 0.0
    return cat_vec + [rating_norm, reviews_norm, price_norm, verified]


def user_preference_vector(interactions: list, businesses: list):
    """
    Build a weighted average feature vector from the user's interaction history.
    """
    biz_map = {str(b["_id"]): b for b in businesses}
    total_weight = 0.0
    vec = np.zeros(len(CATEGORIES) + 4)

    for inter in interactions:
        biz = biz_map.get(inter["businessId"])
        if biz:
            fv = np.array(business_feature_vector(biz))
            vec += fv * inter["weight"]
            total_weight += inter["weight"]

    if total_weight > 0:
        vec /= total_weight
    return vec


# ── Reason generator ──
def generate_reason(biz, score, user_lat, user_lng):
    biz_coords = biz.get("location", {}).get("coordinates", [0, 0])
    dist = haversine(user_lat, user_lng, biz_coords[1], biz_coords[0])

    if dist < 1.0:
        return "Right next to you"
    if biz.get("rating", 0) >= 4.7:
        return "Exceptionally rated"
    if biz.get("isVerified"):
        return "Verified & trusted"
    if biz.get("totalReviews", 0) > 200:
        return "Popular in your area"
    if score > 0.85:
        return "Matches your preferences"
    return "Recommended for you"


@app.get("/health")
def health():
    return {"status": "ok", "service": "LocalBiz ML", "timestamp": datetime.utcnow().isoformat()}


@app.get("/recommend")
def recommend(
    userId: Optional[str] = Query(None),
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    limit: int = Query(10, le=30),
    maxDistance: float = Query(15.0, description="Max distance in km"),
):
    businesses = load_businesses()
    if not businesses:
        raise HTTPException(status_code=503, detail="No business data available")

    # ── Step 1: Filter by distance ──
    nearby = []
    for b in businesses:
        coords = b.get("location", {}).get("coordinates", [0, 0])
        dist = haversine(lat, lng, coords[1], coords[0])
        if dist <= maxDistance:
            nearby.append({**b, "_dist": dist})

    if not nearby:
        nearby = [{**b, "_dist": 0} for b in businesses[:20]]

    # ── Step 2: Build feature matrix ──
    feature_matrix = np.array([business_feature_vector(b) for b in nearby])

    # ── Step 3: Get user preference vector ──
    interactions = []
    content_score = np.ones(len(nearby)) * 0.5  # default: neutral

    if userId:
        try:
            from bson import ObjectId
            interactions = load_user_interactions(ObjectId(userId))
        except Exception:
            pass

    if interactions:
        user_vec = user_preference_vector(interactions, businesses)
        # Cosine similarity between user preference and each business
        if np.any(user_vec):
            sims = cosine_similarity([user_vec], feature_matrix)[0]
            content_score = sims
    else:
        # Cold start: use rating + review count as proxy
        ratings = np.array([b.get("rating", 0) or 0 for b in nearby]) / 5.0
        reviews = np.array([min(b.get("totalReviews", 0) or 0, 500) for b in nearby]) / 500.0
        content_score = 0.6 * ratings + 0.4 * reviews

    # ── Step 4: Distance score (closer = higher) ──
    max_dist = max(b["_dist"] for b in nearby) or 1
    dist_score = np.array([1 - (b["_dist"] / max_dist) for b in nearby])

    # ── Step 5: Popularity score ──
    max_reviews = max(b.get("totalReviews", 0) or 0 for b in nearby) or 1
    pop_score = np.array([min(b.get("totalReviews", 0) or 0, max_reviews) / max_reviews for b in nearby])

    # ── Step 6: Final weighted score ──
    # Personalised: content=0.45, distance=0.35, popularity=0.20
    # Cold-start: content=0.30, distance=0.45, popularity=0.25
    w_content = 0.45 if interactions else 0.30
    w_dist = 0.35 if interactions else 0.45
    w_pop = 0.20 if interactions else 0.25

    final_score = w_content * content_score + w_dist * dist_score + w_pop * pop_score

    # ── Step 7: Rank and return top N ──
    # Exclude businesses user already interacted with
    interacted_ids = {i["businessId"] for i in interactions}
    results = []
    for i, biz in enumerate(nearby):
        bid = str(biz["_id"])
        if bid in interacted_ids:
            continue
        results.append({
            "businessId": bid,
            "score": round(float(final_score[i]), 4),
            "reason": generate_reason(biz, final_score[i], lat, lng),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"success": True, "recommendations": results[:limit]}


@app.get("/similar/{business_id}")
def similar_businesses(
    business_id: str,
    lat: float = Query(...),
    lng: float = Query(...),
    limit: int = Query(5, le=20),
):
    """Find businesses similar to a given one (content-based)."""
    businesses = load_businesses()
    target = next((b for b in businesses if str(b["_id"]) == business_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Business not found")

    target_vec = np.array(business_feature_vector(target)).reshape(1, -1)
    feature_matrix = np.array([business_feature_vector(b) for b in businesses])
    similarities = cosine_similarity(target_vec, feature_matrix)[0]

    results = []
    for i, biz in enumerate(businesses):
        if str(biz["_id"]) == business_id:
            continue
        coords = biz.get("location", {}).get("coordinates", [0, 0])
        dist = haversine(lat, lng, coords[1], coords[0])
        results.append({
            "businessId": str(biz["_id"]),
            "score": round(float(similarities[i]), 4),
            "distance": round(dist, 2),
            "reason": "Similar to what you're viewing",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"success": True, "similar": results[:limit]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
