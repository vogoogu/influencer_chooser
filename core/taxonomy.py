"""Niche taxonomy, brand categories and bilingual keywords.

The keyword lists serve two purposes:
1. rule-based brief analysis when no AI key is available,
2. a cross-lingual bridge for TF-IDF embeddings (briefs are often English, bios German).

CATEGORY_NICHES must stay in sync with generate_dataset.py (a test checks this).
"""
from __future__ import annotations

NICHE_KEYWORDS: dict[str, list[str]] = {
    "vegan_cooking": ["vegan", "plant-based", "plant based", "pflanzlich", "vegane", "tofu", "meal prep", "protein bowl", "vegetarisch", "ready meal", "fertiggericht"],
    "baking": ["baking", "bake", "backen", "kuchen", "bread", "brot", "strudel", "dessert", "pastry", "snack"],
    "coffee_tea": ["coffee", "kaffee", "tea", "tee", "matcha", "espresso", "barista", "roast"],
    "running": ["running", "runner", "laufen", "laufschuh", "trail", "marathon", "run shoe", "running shoe"],
    "yoga_pilates": ["yoga", "pilates", "mobility", "stretching", "mindfulness"],
    "strength_training": ["strength", "gym", "krafttraining", "workout", "protein powder", "fitness"],
    "nutrition": ["nutrition", "ernährung", "diet", "healthy eating", "vitamin", "gesund", "healthy"],
    "skincare": ["skincare", "hautpflege", "serum", "cosmetic", "kosmetik", "sunscreen", "beauty"],
    "sustainable_fashion": ["sustainable fashion", "fair fashion", "second-hand", "secondhand", "nachhaltige mode", "recycled", "upcycling", "slow fashion"],
    "streetwear": ["streetwear", "sneaker", "urban fashion", "fashion", "mode"],
    "gadgets_tech": ["tech", "gadget", "smartphone", "headphone", "kopfhörer", "device", "elektronik", "electronics"],
    "gaming": ["gaming", "game", "games", "console", "esports", "spiel"],
    "hiking_alpine": ["hiking", "wandern", "alpine", "mountain", "berg", "outdoor", "trekking"],
    "city_travel": ["travel", "reise", "citytrip", "städtereise", "hotel", "tourism", "urlaub"],
    "skiing_snowboard": ["ski", "snowboard", "winter sports", "wintersport", "skitour"],
    "parenting": ["parenting", "baby", "kids", "eltern", "familie", "family", "toddler", "kleinkind"],
    "interior_design": ["interior", "wohnen", "furniture", "möbel", "home decor", "einrichtung"],
    "diy": ["diy", "selber machen", "selbermachen", "tools", "werkzeug", "craft", "basteln"],
    "personal_finance": ["finance", "finanz", "banking", "bank", "savings", "sparen", "etf", "invest", "money", "konto", "first-time savers"],
    "career_tips": ["career", "karriere", "job", "bewerbung", "cv", "lebenslauf"],
    "dogs_cats": ["dog", "cat", "pet", "hund", "katze", "haustier", "tierfutter"],
}

NICHES = list(NICHE_KEYWORDS)

NICHE_LABEL_DE = {
    "vegan_cooking": "Vegane Küche", "baking": "Backen", "coffee_tea": "Kaffee und Tee", "running": "Laufen",
    "yoga_pilates": "Yoga und Pilates", "strength_training": "Krafttraining", "nutrition": "Ernährung",
    "skincare": "Hautpflege", "sustainable_fashion": "Nachhaltige Mode", "streetwear": "Streetwear",
    "gadgets_tech": "Technik und Gadgets", "gaming": "Gaming", "hiking_alpine": "Wandern in den Alpen",
    "city_travel": "Städtereisen", "skiing_snowboard": "Ski und Snowboard", "parenting": "Familienalltag",
    "interior_design": "Wohnen und Interior", "diy": "Selbermachen", "personal_finance": "Finanzen",
    "career_tips": "Karrieretipps", "dogs_cats": "Hunde und Katzen",
}

PARENT_OF = {
    "vegan_cooking": "Food & Drink", "baking": "Food & Drink", "coffee_tea": "Food & Drink",
    "running": "Fitness & Health", "yoga_pilates": "Fitness & Health", "strength_training": "Fitness & Health",
    "nutrition": "Fitness & Health", "skincare": "Beauty & Fashion", "sustainable_fashion": "Beauty & Fashion",
    "streetwear": "Beauty & Fashion", "gadgets_tech": "Tech & Gaming", "gaming": "Tech & Gaming",
    "hiking_alpine": "Travel & Outdoor", "city_travel": "Travel & Outdoor", "skiing_snowboard": "Travel & Outdoor",
    "parenting": "Family & Home", "interior_design": "Family & Home", "diy": "Family & Home",
    "personal_finance": "Finance & Career", "career_tips": "Finance & Career", "dogs_cats": "Pets",
}

CATEGORY_NICHES = {
    "plant_based_food": ["vegan_cooking", "nutrition"], "bakery_snacks": ["baking", "vegan_cooking"],
    "coffee_beverages": ["coffee_tea"], "sportswear": ["running", "yoga_pilates", "strength_training", "skiing_snowboard"],
    "fitness_nutrition": ["strength_training", "nutrition", "running"], "skincare": ["skincare"],
    "fashion": ["sustainable_fashion", "streetwear"], "tech_gadgets": ["gadgets_tech", "gaming"],
    "gaming": ["gaming"], "travel": ["city_travel", "hiking_alpine"], "outdoor_gear": ["hiking_alpine", "skiing_snowboard"],
    "fintech": ["personal_finance", "career_tips"], "pet_food": ["dogs_cats"],
    "home_interior": ["interior_design", "diy"], "baby_family": ["parenting"],
    "alcoholic_beverages": ["city_travel", "streetwear"],
}
CATEGORIES = list(CATEGORY_NICHES)
CATEGORY_LABELS = {c: c.replace("_", " ") for c in CATEGORIES}

# category specific words that decide between categories sharing niches (e.g. sportswear vs fitness)
CATEGORY_KEYWORDS = {
    "plant_based_food": ["plant-based", "vegan", "ready meal", "protein bowl", "food", "bowl", "supermarket", "supermarkt"],
    "bakery_snacks": ["bakery", "snack", "bake", "bread", "cake"],
    "coffee_beverages": ["coffee", "tea", "kaffee", "beverage", "drink"],
    "sportswear": ["shoe", "sneaker", "apparel", "sportswear", "running shoe", "trail", "schuh"],
    "fitness_nutrition": ["supplement", "protein powder", "nutrition", "shake"],
    "skincare": ["skincare", "serum", "cream", "cosmetic", "beauty"],
    "fashion": ["fashion", "clothing", "label", "kleidung"],
    "tech_gadgets": ["gadget", "device", "smartphone", "headphone", "tech"],
    "gaming": ["game", "gaming", "console"],
    "travel": ["travel", "trip", "hotel", "booking", "reise"],
    "outdoor_gear": ["outdoor", "gear", "backpack", "ski", "snowboard"],
    "fintech": ["banking", "bank", "app", "savings", "finance", "account", "konto", "sparen"],
    "pet_food": ["pet food", "dog food", "cat food", "tierfutter"],
    "home_interior": ["furniture", "interior", "home", "möbel"],
    "baby_family": ["baby", "diaper", "windel", "kids"],
    "alcoholic_beverages": ["beer", "wine", "spirits", "gin", "bier", "wein", "alcohol"],
}

COUNTRY_WORDS = {
    "AT": ["austria", "österreich", "oesterreich", "vienna", "wien", "graz", "salzburg", "innsbruck", "linz"],
    "DE": ["germany", "deutschland", "berlin", "münchen", "munich", "hamburg"],
    "CH": ["switzerland", "schweiz", "zürich", "zurich", "basel", "bern"],
    "IT": ["italy", "italien", "italia"], "GB": ["united kingdom", "uk", "england", "britain"],
}
FLAG_WORDS = {
    "alcohol_promotion": ["alcohol", "alkohol", "beer", "wine", "spirits"],
    "gambling_promotion": ["gambling", "glücksspiel", "betting", "casino"],
    "crypto_promotion": ["crypto", "krypto", "bitcoin"],
    "political_content": ["political", "politik", "politisch"],
}
GOAL_WORDS = {
    "app_installs": ["install", "download", "app store"],
    "sales": ["sales", "sell", "purchase", "verkauf", "conversion", "launch"],
    "traffic": ["traffic", "clicks", "website visits", "klicks"],
    "awareness": ["awareness", "reach", "bekanntheit", "visibility"],
}


def niche_expansion(niches: list[str]) -> str:
    """Bilingual text for a list of niches, used to bridge DE/EN in embeddings."""
    parts: list[str] = []
    for n in niches:
        if n in NICHE_KEYWORDS:
            parts.append(n.replace("_", " "))
            parts.append(NICHE_LABEL_DE.get(n, ""))
            parts.extend(NICHE_KEYWORDS[n])
    return " ".join(p for p in parts if p)
