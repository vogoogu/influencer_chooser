#!/usr/bin/env python3
"""
Synthetic influencer database generator (DACH market focus).

Everything produced here is FICTIONAL. Names, handles, brands and numbers are random;
any resemblance to real people or companies is coincidental. The *distributions* are
calibrated to public 2026 benchmarks (see DATA_SOURCES.md) so the data behaves
realistically, e.g. engagement falls and prices rise with audience size.

Usage:
    python generate_dataset.py                 # writes CSVs + SQLite to ./data
    python generate_dataset.py --seed 7 --n 120

The run is fully deterministic for a given seed.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TODAY = date(2026, 10, 9)          # "now" of the dataset
N_MONTHS = 18                      # monthly snapshots: 2025-04-30 .. 2026-09-30
SPIKE_IDX = 14                     # month index (2026-06-30) where fake-follower spikes occur
STALE_LAST_IDX = 11                # stale accounts stop at 2026-03-31

# ======================================================================
# CALIBRATION BLOCK - edit here to change the "world". Sources: DATA_SOURCES.md
# Sources disagree by 2-4x on engagement; values below sit in the middle.
# ======================================================================
TIERS = [("nano", 1_000, 10_000), ("micro", 10_000, 100_000), ("mid", 100_000, 500_000),
         ("macro", 500_000, 1_000_000), ("mega", 1_000_000, 2_500_000)]
TIER_PROBS = [0.33, 0.40, 0.15, 0.09, 0.03]

# median engagement rate, % of followers (likes+comments+shares+saves)
BENCH_ER = {
    "instagram": {"nano": 3.2, "micro": 1.5, "mid": 1.0, "macro": 0.65, "mega": 0.40},
    "tiktok":    {"nano": 5.0, "micro": 3.5, "mid": 2.2, "macro": 1.5,  "mega": 1.0},
    "youtube":   {"nano": 3.0, "micro": 1.2, "mid": 0.6, "macro": 0.35, "mega": 0.20},
}
# average views per post as a share of followers
REACH_FACTOR = {
    "instagram": {"nano": 0.45, "micro": 0.30, "mid": 0.20, "macro": 0.14, "mega": 0.10},
    "tiktok":    {"nano": 0.60, "micro": 0.45, "mid": 0.30, "macro": 0.20, "mega": 0.14},
    "youtube":   {"nano": 0.10, "micro": 0.07, "mid": 0.05, "macro": 0.035, "mega": 0.025},
}
MONTHLY_GROWTH = {"nano": 0.025, "micro": 0.015, "mid": 0.009, "macro": 0.006, "mega": 0.004}
FAKE_PCT_MEDIAN = {"nano": 4.0, "micro": 6.0, "mid": 9.0, "macro": 12.0, "mega": 15.0}
HOME_AUDIENCE_SHARE = {"nano": 0.78, "micro": 0.70, "mid": 0.58, "macro": 0.48, "mega": 0.40}
VERIFIED_PROB = {"nano": 0.0, "micro": 0.03, "mid": 0.15, "macro": 0.40, "mega": 0.70}
SPONSORED_SHARE = {"nano": 0.12, "micro": 0.22, "mid": 0.30, "macro": 0.38, "mega": 0.45}

# Price for a feed post in EUR ~ PRICE_C * followers ** PRICE_B  (fitted to DACH 2026 ranges)
PRICE_C, PRICE_B = 0.1205, 0.82
DELIVERABLE_MULT = {"feed_post": 1.0, "reel": 1.8, "story_set": 0.7, "video": 1.1}

# The naive prediction model used for the historic campaigns is optimistic by tier
PREDICTION_BIAS = {"nano": -0.05, "micro": -0.10, "mid": -0.15, "macro": -0.22, "mega": -0.28}
MODEL_VERSION = "naive-v1"

# ======================================================================
# Taxonomy, niches, brands
# ======================================================================
NICHE_TREE = {
    "Food & Drink": ["vegan_cooking", "baking", "coffee_tea"],
    "Fitness & Health": ["running", "yoga_pilates", "strength_training", "nutrition"],
    "Beauty & Fashion": ["skincare", "sustainable_fashion", "streetwear"],
    "Tech & Gaming": ["gadgets_tech", "gaming"],
    "Travel & Outdoor": ["hiking_alpine", "city_travel", "skiing_snowboard"],
    "Family & Home": ["parenting", "interior_design", "diy"],
    "Finance & Career": ["personal_finance", "career_tips"],
    "Pets": ["dogs_cats"],
}
PARENT_OF = {leaf: p for p, leaves in NICHE_TREE.items() for leaf in leaves}
HANDLE_SUFFIX = {"Food & Drink": "kocht", "Fitness & Health": "moves", "Beauty & Fashion": "style",
                 "Tech & Gaming": "tech", "Travel & Outdoor": "unterwegs", "Family & Home": "home",
                 "Finance & Career": "money", "Pets": "pets"}

# leaf: (female share of audience, price multiplier, sampling weight, DE subjects, EN subjects, hashtags)
NICHE_SPEC = {
    "vegan_cooking": (0.72, 1.00, 8, ["Linsen-Curry", "Kichererbsen-Bowl", "Hafermilch-Pancakes", "Tofu-Scramble", "Meal-Prep für die Woche"],
                      ["lentil curry", "chickpea bowl", "oat pancakes", "tofu scramble", "weekly meal prep"], ["vegan", "plantbased", "rezepte"]),
    "baking": (0.82, 0.95, 5, ["Sauerteigbrot", "Marillenknödel", "Apfelstrudel", "Käsekuchen", "Zimtschnecken"],
               ["sourdough loaf", "apricot dumplings", "apple strudel", "cheesecake", "cinnamon rolls"], ["backen", "homemade", "sweettooth"]),
    "coffee_tea": (0.55, 1.00, 3, ["Filterkaffee zuhause", "Specialty-Rösterei", "Matcha-Ritual", "Cold Brew", "Café-Tipps"],
                   ["pour-over at home", "specialty roasters", "matcha ritual", "cold brew", "cafe guide"], ["coffee", "specialtycoffee", "teatime"]),
    "running": (0.50, 1.00, 6, ["Intervalltraining", "Halbmarathon-Vorbereitung", "Trailrunning-Route", "Laufschuh-Test", "Regenerationslauf"],
                ["interval session", "half marathon prep", "trail route", "running shoe test", "recovery run"], ["running", "laufen", "trailrun"]),
    "yoga_pilates": (0.88, 1.00, 4, ["Morgen-Flow", "Pilates für den Rücken", "Atemübungen", "Yoga im Büro", "Mobility-Routine"],
                     ["morning flow", "pilates for back pain", "breathing exercises", "desk yoga", "mobility routine"], ["yoga", "pilates", "mindfulness"]),
    "strength_training": (0.38, 1.05, 5, ["Push-Pull-Split", "Kniebeugen-Technik", "Home-Workout", "Proteinreiche Rezepte", "Trainingsplan"],
                          ["push-pull split", "squat technique", "home workout", "high-protein recipes", "training plan"], ["gym", "strength", "fitness"]),
    "nutrition": (0.68, 1.10, 4, ["Ballaststoffe im Alltag", "Frühstücksideen", "Mythen über Zucker", "Einkaufsliste", "Mikronährstoffe"],
                  ["fibre in daily life", "breakfast ideas", "sugar myths", "grocery list", "micronutrients"], ["ernährung", "healthyeating", "nutrition"]),
    "skincare": (0.90, 1.15, 6, ["Abendroutine", "Sonnenschutz im Winter", "Retinol für Anfänger", "Hautbarriere", "Drogerie-Favoriten"],
                 ["evening routine", "winter sun protection", "retinol for beginners", "skin barrier", "drugstore favourites"], ["skincare", "hautpflege", "glowup"]),
    "sustainable_fashion": (0.82, 1.05, 4, ["Second-Hand-Fund", "Capsule Wardrobe", "Upcycling", "Fair-Fashion-Label", "Pflege von Wolle"],
                            ["second-hand find", "capsule wardrobe", "upcycling", "fair fashion label", "wool care"], ["slowfashion", "secondhand", "ootd"]),
    "streetwear": (0.42, 1.00, 4, ["Sneaker-Drop", "Layering im Herbst", "Vintage-Jacke", "Outfit des Tages", "Thrift-Haul"],
                   ["sneaker drop", "autumn layering", "vintage jacket", "outfit of the day", "thrift haul"], ["streetwear", "sneakers", "fit"]),
    "gadgets_tech": (0.25, 1.20, 4, ["Smartphone-Kamera", "Kopfhörer im Test", "Smart-Home-Setup", "Powerbank-Vergleich", "Tablet für Studierende"],
                     ["smartphone camera", "headphone review", "smart home setup", "power bank comparison", "tablet for students"], ["tech", "gadgets", "review"]),
    "gaming": (0.22, 1.00, 4, ["Indie-Spiel der Woche", "Controller-Setup", "Speedrun", "Retro-Konsole", "Co-op-Abend"],
               ["indie game of the week", "controller setup", "speedrun", "retro console", "co-op night"], ["gaming", "gamer", "indiegame"]),
    "hiking_alpine": (0.50, 1.00, 5, ["Gipfeltour", "Hüttenwanderung", "Packliste", "Sonnenaufgang am Berg", "Klettersteig"],
                      ["summit tour", "hut-to-hut hike", "packing list", "sunrise on the mountain", "via ferrata"], ["wandern", "berge", "hiking"]),
    "city_travel": (0.62, 1.05, 5, ["Wochenende in Lissabon", "Interrail-Route", "Geheimtipps Wien", "Handgepäck-Tricks", "Café-Hopping"],
                    ["weekend in Lisbon", "interrail route", "Vienna hidden gems", "carry-on tricks", "cafe hopping"], ["reisen", "citytrip", "travel"]),
    "skiing_snowboard": (0.40, 1.00, 3, ["Pulverschnee-Tag", "Skitourengehen", "Ausrüstungs-Check", "Après-Ski-Tipps", "Saisonstart"],
                         ["powder day", "ski touring", "gear check", "après-ski tips", "season opening"], ["ski", "snowboard", "wintersport"]),
    "parenting": (0.90, 0.95, 5, ["Schlafroutine", "Beikost-Start", "Ausflug mit Kleinkind", "Ordnung im Kinderzimmer", "Elternalltag"],
                  ["sleep routine", "starting solids", "trip with a toddler", "tidy kids room", "parent life"], ["mamaalltag", "familie", "parenting"]),
    "interior_design": (0.78, 1.00, 4, ["kleine Wohnung", "Naturmaterialien", "Flohmarkt-Fund", "Farbkonzept", "Regal-Styling"],
                        ["small apartment", "natural materials", "flea market find", "colour concept", "shelf styling"], ["interior", "wohnen", "homedecor"]),
    "diy": (0.50, 0.95, 3, ["Holzregal selber bauen", "Upcycling-Möbel", "Wandfarbe", "Pflanzenampel", "Werkzeug-Basics"],
            ["build a wooden shelf", "upcycled furniture", "wall paint", "plant hanger", "tool basics"], ["diy", "selbermachen", "upcycling"]),
    "personal_finance": (0.38, 1.50, 4, ["ETF-Sparplan", "Budget-Methode", "Notgroschen", "Steuer-Tipps", "Gehaltsverhandlung"],
                         ["ETF savings plan", "budgeting method", "emergency fund", "tax tips", "salary negotiation"], ["finanzen", "etf", "moneytips"]),
    "career_tips": (0.55, 1.20, 2, ["Bewerbungsgespräch", "Lebenslauf-Check", "Homeoffice-Routine", "Karrierewechsel", "Networking"],
                    ["job interview", "CV check", "home office routine", "career change", "networking"], ["karriere", "jobtipps", "career"]),
    "dogs_cats": (0.74, 0.95, 4, ["Welpentraining", "Katzenfutter im Vergleich", "Gassi-Route", "Tierarzt-Check", "Spielzeug-Test"],
                  ["puppy training", "cat food comparison", "dog walk route", "vet check", "toy test"], ["hund", "katze", "petlife"]),
}
NICHE_TONES = {
    "personal_finance": ["educational", "authentic_raw"], "career_tips": ["educational", "inspirational"],
    "skincare": ["premium", "educational"], "gaming": ["humorous", "casual"], "streetwear": ["casual", "premium"],
    "parenting": ["authentic_raw", "casual"], "gadgets_tech": ["educational", "casual"],
    "hiking_alpine": ["inspirational", "authentic_raw"], "yoga_pilates": ["inspirational", "premium"],
}
NICHE_LABEL_DE = {
    "vegan_cooking": "Vegane Küche", "baking": "Backen", "coffee_tea": "Kaffee & Tee", "running": "Laufen",
    "yoga_pilates": "Yoga & Pilates", "strength_training": "Krafttraining", "nutrition": "Ernährung",
    "skincare": "Hautpflege", "sustainable_fashion": "Nachhaltige Mode", "streetwear": "Streetwear",
    "gadgets_tech": "Technik & Gadgets", "gaming": "Gaming", "hiking_alpine": "Wandern in den Alpen",
    "city_travel": "Städtereisen", "skiing_snowboard": "Ski & Snowboard", "parenting": "Familienalltag",
    "interior_design": "Wohnen & Interior", "diy": "Selbermachen", "personal_finance": "Finanzen",
    "career_tips": "Karrieretipps", "dogs_cats": "Hunde & Katzen",
}
ALL_TONES = ["casual", "humorous", "educational", "inspirational", "premium", "authentic_raw"]

# brand category -> niches that fit
CATEGORY_BRANDS = {
    "plant_based_food": ["Gruenwerk Foods", "Veggora", "Kornblume Kitchen"],
    "bakery_snacks": ["Mehlkind", "Crumbly Joe", "Backstube Nord"],
    "coffee_beverages": ["Bergbohne Coffee", "Teekontor Alpin", "Kaffeeraum"],
    "sportswear": ["Nordstride", "Alpenflex", "Velocita Sport"],
    "fitness_nutrition": ["Muskelkorn", "ProtiPeak", "FitFuel Lab"],
    "skincare": ["Hautnah Labs", "Lumera Skin", "Reinblatt Cosmetics"],
    "fashion": ["Faden & Form", "Urbanloom", "Kreislauf Mode"],
    "tech_gadgets": ["Pixelbrook", "Zentrix", "Nodewave"],
    "gaming": ["Questforge", "Pixelhorn Games"],
    "travel": ["Wegwerk Reisen", "Citylyst"],
    "outdoor_gear": ["Gipfelwerk", "Schneehaus Outdoor"],
    "fintech": ["Kontoro", "Sparfuchs App", "Fiscalo"],
    "pet_food": ["Pfotenglueck", "Napfwerk"],
    "home_interior": ["Wohnraum Atelier", "Holzart Home"],
    "baby_family": ["Kleinstern", "Windelwunder"],
    "alcoholic_beverages": ["Alpenbrenner", "Hopfenhain"],
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
NICHE_CATEGORIES: dict[str, list[str]] = {}
for _cat, _ns in CATEGORY_NICHES.items():
    for _n in _ns:
        NICHE_CATEGORIES.setdefault(_n, []).append(_cat)

# ======================================================================
# Names, places, text
# ======================================================================
COUNTRIES = ["AT", "DE", "CH", "GB", "IT", "CZ", "HU"]
COUNTRY_P = [0.45, 0.30, 0.10, 0.05, 0.05, 0.03, 0.02]
CITIES = {"AT": ["Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt", "Wiener Neustadt", "St. Pölten", "Bregenz", "Villach"],
          "DE": ["Berlin", "München", "Hamburg", "Köln", "Frankfurt", "Stuttgart", "Leipzig"],
          "CH": ["Zürich", "Bern", "Basel", "Luzern"], "GB": ["London", "Manchester", "Bristol"],
          "IT": ["Milano", "Bolzano", "Roma"], "CZ": ["Praha", "Brno"], "HU": ["Budapest"]}
LANGS = {"AT": "de,en", "DE": "de,en", "CH": "de,en", "GB": "en", "IT": "it,en", "CZ": "cs,en", "HU": "hu,en"}
OTHER_AUDIENCE = {"AT": ["DE", "CH", "IT", "HU"], "DE": ["AT", "CH", "NL", "PL"], "CH": ["DE", "AT", "FR", "IT"],
                  "GB": ["US", "IE", "DE", "AU"], "IT": ["AT", "CH", "DE", "FR"], "CZ": ["SK", "DE", "AT", "PL"],
                  "HU": ["SK", "AT", "DE", "RO"]}
NAMES = {
    "dach": (["Lena", "Anna", "Sophie", "Julia", "Laura", "Mia", "Hannah", "Marie", "Lisa", "Katharina", "Theresa", "Valentina",
              "Nora", "Johanna", "Elena", "Magdalena", "Sarah", "Franziska", "Clara", "Emilia",
              "Lukas", "Jonas", "Felix", "Maximilian", "Paul", "Tobias", "David", "Florian", "Elias", "Moritz", "Daniel",
              "Matthias", "Philipp", "Stefan", "Noah", "Fabian", "Simon", "Jakob", "Leon", "Raphael"],
             ["Hofer", "Gruber", "Huber", "Bauer", "Wagner", "Müller", "Pichler", "Steiner", "Moser", "Mayer", "Berger", "Fischer",
              "Eder", "Leitner", "Fuchs", "Weber", "Schmid", "Winkler", "Reiter", "Brunner", "Lang", "Haas", "Koller", "Auer",
              "Egger", "Kraus", "Hartmann", "Neumann", "Vogel", "Lindner"]),
    "GB": (["Emma", "Oliver", "Chloe", "Jack", "Grace", "Harry", "Ruby", "Leo"], ["Walker", "Bennett", "Hughes", "Collins", "Foster", "Ellis"]),
    "IT": (["Giulia", "Marco", "Chiara", "Luca", "Sofia", "Matteo"], ["Rossi", "Bianchi", "Conti", "Ferrari", "Romano"]),
    "CZ": (["Tereza", "Jakub", "Eliska", "Ondrej"], ["Novak", "Dvorak", "Svoboda", "Cerny"]),
    "HU": (["Anna", "Bence", "Reka", "Mate"], ["Nagy", "Kovacs", "Toth", "Szabo"]),
}
DE_ORGANIC = ["Heute: {s}. Was meint ihr?", "Neu auf dem Kanal: {s}.", "{s} - drei Dinge, die ich gelernt habe.",
              "Mein Wochenende mit {s}.", "Frage an euch: Wie haltet ihr es mit {s}?"]
EN_ORGANIC = ["Today: {s}. Thoughts?", "New this week: {s}.", "{s} - three things I learned.",
              "My weekend with {s}.", "Question for you: how do you handle {s}?"]
DE_SPONSORED = ["Werbung | In Zusammenarbeit mit {b}: {s}.", "Anzeige - {b} hat mich bei {s} unterstützt.", "#Werbung {b} und {s}: mein ehrlicher Eindruck."]
EN_SPONSORED = ["Ad | In partnership with {b}: {s}.", "Sponsored - {b} supported my {s} this week.", "#ad {b} and {s}: my honest take."]
NOTES_BY_RELIABILITY = [
    (0.90, ["Reliable, usually delivers early.", "Great communication, easy to brief.", "Strong community with high comment quality."]),
    (0.75, ["Solid partner, needs a clear brief.", "Prefers reels over static posts.", "Needs about two weeks lead time."]),
    (0.00, ["Late with delivery in past campaigns.", "Responds slowly, follow up twice.", "Negotiates hard on usage rights."]),
]
FLAG_TYPES = {"alcohol_promotion": (1, 2), "gambling_promotion": (3, 3), "political_content": (2, 2), "controversy": (3, 3),
              "undisclosed_advertising": (2, 2), "crypto_promotion": (3, 3), "profanity": (1, 1)}
FLAG_P = [0.20, 0.10, 0.20, 0.15, 0.15, 0.10, 0.10]
FLAG_SOURCES = ["manual_review", "news_check", "community_report"]

PLATFORM_URL = {"instagram": "https://www.instagram.com/{h}", "tiktok": "https://www.tiktok.com/@{h}",
                "youtube": "https://www.youtube.com/@{h}"}
FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", " ": ""})


# ======================================================================
# helpers
# ======================================================================
def month_ends() -> list[date]:
    out, y, m = [], 2025, 4
    for _ in range(N_MONTHS):
        nxt = date(y + (m // 12), m % 12 + 1, 1)
        out.append(nxt - timedelta(days=1))
        y, m = nxt.year, nxt.month
    return out


MONTHS = month_ends()


def month_idx(d: date) -> int:
    """Index of the last month-end snapshot strictly before the given date."""
    i = (d.year - 2025) * 12 + d.month - 4 - 1
    return int(min(max(i, 0), N_MONTHS - 1))


def tier_of(followers: float) -> str:
    for name, lo, hi in TIERS:
        if followers < hi:
            return name
    return "mega"


def iso(d: date) -> str:
    return d.isoformat()


class Gen:
    def __init__(self, seed: int, n: int):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.t: dict[str, list[dict]] = {k: [] for k in [
            "niche", "influencer", "influencer_niche", "account", "account_snapshot", "audience_snapshot",
            "brand", "brand_competitor", "content_item", "rate_card", "safety_flag", "exclusivity",
            "campaign", "participation", "result"]}
        self.truth: list[dict] = []
        self.ids = {k: 0 for k in self.t}
        self.inf: dict[int, dict] = {}        # influencer_id -> internal info
        self.acc: dict[int, dict] = {}        # account_id -> internal info (series etc.)
        self.brand_id: dict[str, int] = {}
        self.brand_cat: dict[int, str] = {}
        self.niche_id: dict[str, int] = {}

    # ---- small utilities
    def nid(self, table: str) -> int:
        self.ids[table] += 1
        return self.ids[table]

    def lognorm(self, median: float, sigma: float) -> float:
        return float(median * np.exp(self.rng.normal(0, sigma)))

    def choice(self, seq, p=None):
        return seq[int(self.rng.choice(len(seq), p=p))]

    def rand_date(self, start: date, end: date) -> date:
        return start + timedelta(days=int(self.rng.integers(0, (end - start).days + 1)))

    def add(self, table: str, **row) -> int:
        if "id" not in row and table not in ("influencer_niche", "brand_competitor", "result_dummy"):
            row = {"id": self.nid(table), **row}
        elif "id" in row:
            self.ids[table] = max(self.ids[table], row["id"])
        self.t[table].append(row)
        return row.get("id", 0)

    # ------------------------------------------------------------ build steps
    def build_taxonomy_and_brands(self):
        for parent, leaves in NICHE_TREE.items():
            pid = self.add("niche", name=parent, parent_id=None)
            self.niche_id[parent] = pid
            for leaf in leaves:
                self.niche_id[leaf] = self.add("niche", name=leaf, parent_id=pid)
        for cat, names in CATEGORY_BRANDS.items():
            for nm in names:
                bid = self.add("brand", name=nm, category=cat, hq_country=self.choice(["AT", "DE", "CH"], p=[0.5, 0.4, 0.1]))
                self.brand_id[nm] = bid
                self.brand_cat[bid] = cat
        by_cat: dict[str, list[int]] = {}
        for bid, cat in self.brand_cat.items():
            by_cat.setdefault(cat, []).append(bid)
        for ids in by_cat.values():
            for a in ids:
                for b in ids:
                    if a != b:
                        self.add("brand_competitor", brand_id=a, competitor_id=b)

    def build_influencers(self):
        leaves = list(NICHE_SPEC)
        w = np.array([NICHE_SPEC[x][2] for x in leaves], float)
        w /= w.sum()
        used = set()
        for _ in range(self.n):
            country = self.choice(COUNTRIES, p=COUNTRY_P)
            pool = NAMES["dach"] if country in ("AT", "DE", "CH") else NAMES[country]
            while True:
                first, last = self.choice(pool[0]), self.choice(pool[1])
                if (first, last) not in used:
                    used.add((first, last))
                    break
            primary = self.choice(leaves, p=w)
            others = [x for x in leaves if x != primary]
            secondary = []
            if self.rng.random() < 0.6:
                same_parent = [x for x in others if PARENT_OF[x] == PARENT_OF[primary]]
                secondary.append(self.choice(same_parent) if same_parent and self.rng.random() < 0.6 else self.choice(others))
            if self.rng.random() < 0.2:
                extra = [x for x in others if x not in secondary]
                secondary.append(self.choice(extra))
            lang = "de" if country in ("AT", "DE", "CH") else "en"
            city = self.choice(CITIES[country])
            subj = NICHE_SPEC[primary][3 if lang == "de" else 4]
            s1, s2 = self.rng.choice(len(subj), 2, replace=False)
            label = primary.replace("_", " ")
            bio = (f"{NICHE_LABEL_DE[primary]} und mehr aus {city}. Zeige euch {subj[s1]} und {subj[s2]}. Kooperationen via DM."
                   if lang == "de" else
                   f"{label.capitalize()} content from {city}. Sharing {subj[s1]} and {subj[s2]}. DMs open for collabs.")
            tone = self.choice(NICHE_TONES.get(primary, ALL_TONES))
            if self.rng.random() < 0.2:
                tone = self.choice(ALL_TONES)
            r = float(self.rng.beta(6, 2))
            rel = int(np.clip(round(1 + 4 * r + self.rng.normal(0, 0.6)), 1, 5))
            note = ""
            for thr, texts in NOTES_BY_RELIABILITY:
                if r >= thr:
                    note = self.choice(texts)
                    break
            status = self.choice(["active", "paused", "offboarded"], p=[0.90, 0.06, 0.04])
            avail = "unavailable" if status == "offboarded" else self.choice(["open", "limited", "unavailable"], p=[0.60, 0.30, 0.10])
            onboarded = self.rand_date(date(2022, 1, 1), date(2026, 8, 31))
            iid = self.add("influencer", display_name=f"{first} {last}", country=country, city=city, languages=LANGS[country],
                           bio_text=bio, tone=tone, status=status, availability=avail, relationship_score=rel,
                           agency_notes=note, onboarded_at=iso(onboarded),
                           consent_basis=self.choice(["contract", "legitimate_interest"], p=[0.8, 0.2]),
                           consent_until=iso(self.rand_date(date(2027, 3, 1), date(2029, 12, 31))), profile_embedding=None)
            self.inf[iid] = dict(first=first, last=last, country=country, lang=lang, primary=primary, secondary=secondary,
                                 reliability=r, status=status, accounts=[], flags=set(), name=f"{first} {last}")
            self.add("influencer_niche", influencer_id=iid, niche_id=self.niche_id[primary], is_primary=1)
            for s in secondary:
                self.add("influencer_niche", influencer_id=iid, niche_id=self.niche_id[s], is_primary=0)

    def build_accounts_and_snapshots(self):
        platforms = ["instagram", "tiktok", "youtube"]
        for iid, info in self.inf.items():
            parent = PARENT_OF[info["primary"]]
            tier_i = int(self.rng.choice(len(TIERS), p=TIER_PROBS))
            _, lo, hi = TIERS[tier_i]
            f_main = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            main = self.choice(platforms, p=[0.60, 0.25, 0.15])
            plan = [(main, f_main)]
            if self.rng.random() < 0.45:
                second = self.choice([p for p in platforms if p != main])
                plan.append((second, float(np.clip(f_main * np.exp(self.rng.normal(np.log(0.6), 0.5)), 1_000, 2_500_000))))
            base_handle = f"{info['first']}_{info['last']}_{HANDLE_SUFFIX[parent]}".translate(FOLD).lower()
            quality = float(self.rng.normal(0, 0.25))
            for j, (platform, f_end) in enumerate(plan):
                handle = base_handle if j == 0 or self.rng.random() < 0.7 else base_handle + "_official"
                tier = tier_of(f_end)
                aid = self.add("account", influencer_id=iid, platform=platform, handle=handle,
                               url=PLATFORM_URL[platform].format(h=handle),
                               verified=int(self.rng.random() < VERIFIED_PROB[tier]))
                info["accounts"].append(aid)
                self.acc[aid] = dict(influencer_id=iid, platform=platform, quality=quality, spike=False, stale=False,
                                     src="api" if platform == "youtube" and self.rng.random() < 0.7
                                     else self.choice(["csv_import", "manual"], p=[0.65, 0.35]))
                self.acc[aid]["series"] = self._series(f_end, tier, platform, quality)

    def _series(self, f_end, tier, platform, quality):
        rng = self.rng
        g = MONTHLY_GROWTH[tier]
        rates = np.clip(rng.normal(g, g * 0.7, N_MONTHS - 1), -0.01, 0.08)
        f = np.empty(N_MONTHS)
        f[-1] = f_end
        for t in range(N_MONTHS - 1, 0, -1):
            f[t - 1] = f[t] / (1 + rates[t - 1])
        er_end = float(np.clip(self.lognorm(BENCH_ER[platform][tier], 0.35) * np.exp(quality), 0.05, 15))
        x = np.zeros(N_MONTHS)
        for t in range(1, N_MONTHS):
            x[t] = 0.6 * x[t - 1] + rng.normal(0, 0.08)
        er = er_end * np.exp(x - x[-1]) * (f / f_end) ** -0.15
        reach = self.lognorm(REACH_FACTOR[platform][tier], 0.35)
        views = f * reach * np.exp(rng.normal(0, 0.08, N_MONTHS))
        fake_end = float(np.clip(self.lognorm(FAKE_PCT_MEDIAN[tier], 0.4), 1, 28))
        fake = fake_end * np.exp(rng.normal(0, 0.05, N_MONTHS))
        ppw = self.lognorm({"instagram": 3.5, "tiktok": 5.0, "youtube": 1.2}[platform], 0.3)
        return dict(f=f, er=er, views=views, fake=fake, ppw=ppw, tier=tier)

    def plant_trapped_accounts(self):
        """Spiked (fake followers) and stale accounts, recorded in ground truth."""
        cands = [a for a, v in self.acc.items() if v["platform"] != "youtube" and v["series"]["tier"] in ("micro", "mid")
                 and self.inf[v["influencer_id"]]["status"] == "active"]
        picked = list(self.rng.choice(cands, 10, replace=False))
        for aid in picked[:5]:
            s = self.acc[aid]["series"]
            mult = float(self.rng.uniform(1.6, 2.2))
            s["f"][SPIKE_IDX:] *= mult
            s["er"][SPIKE_IDX:] *= 0.45
            s["views"][SPIKE_IDX:] *= 1.5          # purchased views inflate the metric the predictor trusts
            s["fake"][SPIKE_IDX:] = float(self.rng.uniform(32, 45))
            self.acc[aid]["spike"] = True
            self.truth.append(dict(trap_type="fake_follower_spike", entity_type="account", entity_id=aid, related_id="",
                                   detail=f"followers x{mult:.2f} at {MONTHS[SPIKE_IDX]}, engagement x0.45, fake share 32-45%"))
        for aid in picked[5:]:
            self.acc[aid]["stale"] = True
            self.truth.append(dict(trap_type="stale_stats", entity_type="account", entity_id=aid, related_id="",
                                   detail=f"last snapshot {MONTHS[STALE_LAST_IDX]}"))

    def emit_snapshots_and_audience(self):
        conf = {"api": 0.95, "csv_import": 0.80, "manual": 0.70, "ai_estimated": 0.40}
        for aid, v in self.acc.items():
            s, last = v["series"], (STALE_LAST_IDX if v["stale"] else N_MONTHS - 1)
            for t in range(last + 1):
                src = "ai_estimated" if self.rng.random() < 0.06 else v["src"]
                self.add("account_snapshot", account_id=aid, captured_at=iso(MONTHS[t]), followers=int(s["f"][t]),
                         avg_views=int(s["views"][t]), engagement_rate=round(float(s["er"][t]), 3),
                         fake_follower_pct=round(float(np.clip(s["fake"][t], 0.5, 60)), 1),
                         posts_per_week=round(float(s["ppw"] * np.exp(self.rng.normal(0, 0.1))), 1),
                         source=src, confidence=conf[src])
            # audience (quarterly: month idx 2, 5, 8, 11, 14, 17)
            inf = self.inf[v["influencer_id"]]
            fem0 = NICHE_SPEC[inf["primary"]][0] + (0.03 if v["platform"] == "tiktok" else 0)
            alpha = {"instagram": [3, 5, 3, 1.5], "tiktok": [6, 4, 1.5, 0.5], "youtube": [3, 4, 3, 2]}[v["platform"]]
            alpha = np.array(alpha, float)
            if inf["primary"] in ("parenting", "personal_finance", "interior_design", "career_tips"):
                alpha *= np.array([0.3, 1.0, 1.8, 2.0])
            if inf["primary"] in ("gaming", "streetwear"):
                alpha *= np.array([2.0, 1.2, 0.5, 0.2])
            base = self.rng.dirichlet(alpha * 30)
            home = float(np.clip(HOME_AUDIENCE_SHARE[s["tier"]] + self.rng.normal(0, 0.06), 0.2, 0.92))
            for t in (2, 5, 8, 11, 14, 17):
                if t > last:
                    continue
                ages = np.round(self.rng.dirichlet(base * 150 + 0.5) * 100, 1)
                ages[-1] = round(100 - ages[:-1].sum(), 1)
                share_home = float(np.clip(home + self.rng.normal(0, 0.02), 0.2, 0.95))
                rest = (1 - share_home) * self.rng.dirichlet([3, 2, 1])
                others = OTHER_AUDIENCE[inf["country"]]
                top = [{"country": inf["country"], "share": round(share_home, 3)}] + \
                      [{"country": others[k], "share": round(float(rest[k]), 3)} for k in range(3)]
                self.add("audience_snapshot", account_id=aid, captured_at=iso(MONTHS[t]),
                         female_pct=round(float(np.clip(fem0 + self.rng.normal(0, 0.04), 0.05, 0.95)) * 100, 1),
                         age_18_24=float(ages[0]), age_25_34=float(ages[1]), age_35_44=float(ages[2]),
                         age_45_plus=float(ages[3]), top_countries=json.dumps(top))

    def snap(self, aid: int, d: date) -> dict:
        v = self.acc[aid]
        s = v["series"]
        last = STALE_LAST_IDX if v["stale"] else N_MONTHS - 1
        t = min(month_idx(d), last)
        return dict(f=float(s["f"][t]), er=float(s["er"][t]), views=float(s["views"][t]), fake=float(s["fake"][t]), t=t)

    def build_rate_cards(self):
        expired_pool = [i for i, x in self.inf.items() if x["status"] == "active"]
        expired = set(int(i) for i in self.rng.choice(expired_pool, 6, replace=False))
        for iid in expired:
            self.truth.append(dict(trap_type="expired_rate_card", entity_type="influencer", entity_id=iid, related_id="",
                                   detail="all rate cards expired 2026-06-30"))
        for iid, info in self.inf.items():
            vf, vt = ("2025-07-01", "2026-06-30") if iid in expired else ("2026-01-01", "2026-12-31")
            mult = NICHE_SPEC[info["primary"]][1]
            for aid in info["accounts"]:
                v = self.acc[aid]
                s = v["series"]
                f, er, views = float(s["f"][-1]), float(s["er"][-1]), float(s["views"][-1])
                er_adj = (er / BENCH_ER[v["platform"]][s["tier"]]) ** 0.3
                base = PRICE_C * f ** PRICE_B * mult * er_adj
                if v["platform"] == "instagram":
                    items = [("feed_post", 1.0), ("reel", DELIVERABLE_MULT["reel"]), ("story_set", DELIVERABLE_MULT["story_set"])]
                elif v["platform"] == "tiktok":
                    items = [("video", DELIVERABLE_MULT["video"])]
                else:
                    integ = views / 1000 * self.lognorm(28, 0.2) * mult
                    items = [("integration", None), ("dedicated_video", None)]
                for name, m in items:
                    if v["platform"] == "youtube":
                        price = integ * (1 if name == "integration" else 2.2)
                    else:
                        price = base * m * self.lognorm(1, 0.2)
                    price = round(price / (10 if price < 1000 else 50)) * (10 if price < 1000 else 50)
                    self.add("rate_card", influencer_id=iid, platform=v["platform"], deliverable=name,
                             price_eur=float(max(price, 30)), usage_rights_months=3 if name != "story_set" else 0,
                             exclusivity_premium_pct=25.0, valid_from=vf, valid_to=vt)

    def assign_flags(self):
        ids = [int(x) for x in self.rng.choice(list(self.inf), 14, replace=False)]
        keys = list(FLAG_TYPES)
        for iid in ids:
            for _ in range(2 if self.rng.random() < 0.2 else 1):
                ft = self.choice(keys, p=FLAG_P)
                lo, hi = FLAG_TYPES[ft]
                self.inf[iid]["flags"].add(ft)
                fd = self.rand_date(date(2025, 10, 1), date(2026, 9, 20))
                self.add("safety_flag", influencer_id=iid, flag_type=ft, severity=int(self.rng.integers(lo, hi + 1)),
                         source=self.choice(FLAG_SOURCES), evidence_url=f"https://example.invalid/evidence/{iid}-{ft}",
                         flagged_at=iso(fd), expires_at=iso(fd + timedelta(days=365)) if self.rng.random() < 0.5 else None)

    def build_consent_traps(self):
        for iid in self.rng.choice(list(self.inf), 3, replace=False):
            row = next(r for r in self.t["influencer"] if r["id"] == int(iid))
            row["consent_until"] = "2026-08-31"
            self.truth.append(dict(trap_type="consent_expired", entity_type="influencer", entity_id=int(iid), related_id="",
                                   detail="GDPR retention date 2026-08-31 already passed"))

    def emit_content(self, per_account: int = 12):
        for aid, v in self.acc.items():
            inf = self.inf[v["influencer_id"]]
            tier = v["series"]["tier"]
            spec = NICHE_SPEC[inf["primary"]]
            subjects = spec[3 if inf["lang"] == "de" else 4]
            fmts = {"instagram": (["reel", "feed_post", "carousel"], [0.5, 0.3, 0.2]),
                    "tiktok": (["video"], [1.0]), "youtube": (["video", "short"], [0.6, 0.4])}[v["platform"]]
            for _ in range(per_account):
                d = self.rand_date(date(2025, 10, 10), date(2026, 10, 8))
                sponsored = self.rng.random() < SPONSORED_SHARE[tier]
                brand_name, disclosed = None, None
                if sponsored:
                    cats = [c for n in [inf["primary"], *inf["secondary"]] for c in NICHE_CATEGORIES.get(n, [])]
                    if cats and self.rng.random() < 0.8:
                        cat = self.choice(cats)
                    else:
                        cat = self.choice([c for c in CATEGORY_BRANDS if c != "alcoholic_beverages" or "alcohol_promotion" in inf["flags"]])
                    if cat == "alcoholic_beverages" and "alcohol_promotion" not in inf["flags"]:
                        cat = "travel"
                    brand_name = self.choice(CATEGORY_BRANDS[cat])
                    disclosed = int(not ("undisclosed_advertising" in inf["flags"] and self.rng.random() < 0.5))
                self._content_row(aid, d, fmts, subjects, inf, sponsored, brand_name, disclosed)
        # consistency for alcohol flags: make sure at least two sponsored alcohol posts exist
        for iid, inf in self.inf.items():
            if "alcohol_promotion" in inf["flags"]:
                aid = inf["accounts"][0]
                v = self.acc[aid]
                subjects = NICHE_SPEC[inf["primary"]][3 if inf["lang"] == "de" else 4]
                for _ in range(2):
                    self._content_row(aid, self.rand_date(date(2026, 1, 1), date(2026, 9, 25)),
                                      (["reel"] if v["platform"] == "instagram" else ["video"], [1.0]), subjects, inf, True,
                                      self.choice(CATEGORY_BRANDS["alcoholic_beverages"]), 1)

    def _content_row(self, aid, d, fmts, subjects, inf, sponsored, brand_name, disclosed):
        v = self.acc[aid]
        sn = self.snap(aid, d)
        fmt = self.choice(fmts[0], p=fmts[1])
        subj = self.choice(subjects)
        if sponsored:
            tpl = self.choice(DE_SPONSORED if inf["lang"] == "de" else EN_SPONSORED)
            caption = tpl.format(b=brand_name, s=subj)
            if not disclosed:
                caption = caption.replace("Werbung | ", "").replace("Anzeige - ", "").replace("#Werbung ", "")
                caption = caption.replace("Ad | ", "").replace("Sponsored - ", "").replace("#ad ", "")
        else:
            caption = self.choice(DE_ORGANIC if inf["lang"] == "de" else EN_ORGANIC).format(s=subj)
        tags = list(NICHE_SPEC[inf["primary"]][5])[: int(self.rng.integers(2, 4))]
        if sponsored and disclosed:
            tags.append("werbung" if inf["lang"] == "de" else "ad")
        views = sn["views"] * self.lognorm(1.0, 0.45) * (0.6 if fmt in ("feed_post", "carousel") else 1.0)
        inter = sn["er"] / 100 * sn["f"] * self.lognorm(1.0, 0.35) * (0.8 if sponsored else 1.0)
        self.add("content_item", account_id=aid, posted_at=iso(d), format=fmt, caption=caption,
                 hashtags=" ".join("#" + x for x in tags), views=int(views), likes=int(inter * 0.9),
                 comments=int(inter * 0.1), is_sponsored=int(sponsored),
                 brand_id=self.brand_id[brand_name] if brand_name else None, is_disclosed=disclosed, embedding=None)

    # ------------------------------------------------------------ campaigns
    def _pick_deliverable(self, aid):
        p = self.acc[aid]["platform"]
        if p == "instagram":
            return self.choice(["reel", "feed_post", "story_set"], p=[0.55, 0.30, 0.15])
        return "video" if p == "tiktok" else self.choice(["integration", "dedicated_video"], p=[0.7, 0.3])

    def _rate(self, iid, aid, deliverable):
        plat = self.acc[aid]["platform"]
        for r in self.t["rate_card"]:
            if r["influencer_id"] == iid and r["platform"] == plat and r["deliverable"] == deliverable:
                return r["price_eur"]
        return 500.0

    def _pick_influencers(self, category, country, budget, k_max=7):
        niches = CATEGORY_NICHES[category]
        ids, w = [], []
        for iid, info in self.inf.items():
            if info["status"] == "offboarded":
                continue
            fit = 8.0 if info["primary"] in niches else (4.0 if any(n in niches for n in info["secondary"]) else 0.4)
            ids.append(iid)
            w.append(fit * (2.0 if info["country"] == country else 1.0))
        w = np.array(w) / np.sum(w)
        order = list(self.rng.choice(ids, size=min(14, len(ids)), replace=False, p=w))
        picked, spend = [], 0.0
        target_n = int(self.rng.integers(2, k_max + 1))
        for iid in order:
            aid = self.choice(self.inf[iid]["accounts"], p=None)
            dl = self._pick_deliverable(aid)
            fee = self._rate(iid, aid, dl) * float(np.clip(self.rng.normal(0.95, 0.08), 0.75, 1.15))
            if spend + fee <= budget * 1.1 or not picked:
                picked.append((int(iid), aid, dl, round(fee / 10) * 10.0))
                spend += fee
            if len(picked) >= target_n:
                break
        return picked

    def _make_campaign(self, brand_name, goal, budget, start, status, country):
        bid = self.brand_id[brand_name]
        end = start + timedelta(days=int(self.rng.integers(14, 46)))
        season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer",
                  8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}[start.month]
        kind = self.choice(["Launch", "Push", "Creator Series", "Seasonal Drive"])
        return self.add("campaign", brand_id=bid, name=f"{brand_name} {season} {start.year} {kind}", goal=goal,
                        budget_eur=float(budget), start_date=iso(start), end_date=iso(end), target_country=country,
                        status=status), bid, end

    def _predict(self, aid, deliverable, start):
        sn = self.snap(aid, start)
        mult = {"feed_post": 0.7, "story_set": 0.45}.get(deliverable, 1.0)
        p50 = sn["views"] * mult
        return int(p50 * 0.65), int(p50), int(p50 * 1.45), sn

    def build_campaigns(self):
        brands = list(self.brand_id)
        goal_p = [0.35, 0.25, 0.25, 0.15]
        n_hist = 60
        for k in range(n_hist):
            if k < 20:   # cluster late so spiked accounts can be hit after the spike
                start = self.rand_date(date(2026, 7, 1), date(2026, 8, 20))
            else:
                start = self.rand_date(date(2025, 5, 5), date(2026, 6, 30))
            brand = self.choice(brands)
            cat = self.brand_cat[self.brand_id[brand]]
            goal = self.choice(["awareness", "traffic", "sales", "app_installs"], p=goal_p)
            if cat in ("fintech", "gaming") and self.rng.random() < 0.6:
                goal = "app_installs"
            budget = float(np.clip(round(self.lognorm(12000, 0.6) / 500) * 500, 3000, 60000))
            country = self.choice(["AT", "DE", "CH", "IT"], p=[0.55, 0.30, 0.10, 0.05])
            cid, bid, end = self._make_campaign(brand, goal, budget, start, "completed", country)
            for iid, aid, dl, fee in self._pick_influencers(cat, country, budget):
                self._participation(cid, goal, iid, aid, dl, fee, start, end, bid, completed=True)
        # force spike accounts into >= 2 late campaigns each
        late = [r for r in self.t["campaign"] if r["start_date"] >= "2026-07-01" and r["status"] == "completed"]
        for aid, v in self.acc.items():
            if not v["spike"]:
                continue
            for r in [late[int(i)] for i in self.rng.choice(len(late), 2, replace=False)]:
                if any(p["campaign_id"] == r["id"] and p["influencer_id"] == v["influencer_id"] for p in self.t["participation"]):
                    continue
                dl = self._pick_deliverable(aid)
                fee = self._rate(v["influencer_id"], aid, dl) * 0.95
                self._participation(r["id"], r["goal"], v["influencer_id"], aid, dl, round(fee / 10) * 10.0,
                                    date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"]), r["brand_id"], completed=True)
        # running campaigns (no results yet)
        for brand, goal, budget, start, country in [("Alpenflex", "awareness", 18000, date(2026, 9, 21), "AT"),
                                                    ("Lumera Skin", "sales", 14000, date(2026, 9, 28), "DE")]:
            cid, bid, end = self._make_campaign(brand, goal, budget, start, "running", country)
            for iid, aid, dl, fee in self._pick_influencers(self.brand_cat[bid], country, budget, 5):
                self._participation(cid, goal, iid, aid, dl, fee, start, end, bid, completed=False)
        self._planned_campaigns()

    def _participation(self, cid, goal, iid, aid, dl, fee, start, end, bid, completed):
        p10, p50, p90, sn = self._predict(aid, dl, start)
        pid = self.add("participation", campaign_id=cid, influencer_id=iid, account_id=aid, deliverable=dl, fee_eur=float(fee),
                       reach_p10=p10, reach_p50=p50, reach_p90=p90, model_version=MODEL_VERSION)
        if not completed:
            return
        info, v = self.inf[iid], self.acc[aid]
        tier = tier_of(sn["f"])
        eff = PREDICTION_BIAS[tier] + (info["reliability"] - 0.8) * 0.6 - sn["fake"] / 100 * 0.6 + float(self.rng.normal(0, 0.30))
        if v["spike"] and sn["t"] >= SPIKE_IDX:
            eff += np.log(0.5)
        actual = max(int(p50 * np.exp(eff)), 50)
        per_view = float(np.clip(sn["er"] / 100 * sn["f"] / max(sn["views"], 1) * self.lognorm(1, 0.3), 0.005, 0.25))
        cats = CATEGORY_NICHES[self.brand_cat[bid]]
        fit = 1.4 if info["primary"] in cats else (1.0 if any(n in cats for n in info["secondary"]) else 0.7)
        conv_rate = {"awareness": 0.0005, "traffic": 0.012, "sales": 0.0035, "app_installs": 0.004}[goal]
        conversions = int(actual * conv_rate * fit * self.lognorm(1, 0.4))
        on_time = int(self.rng.random() < info["reliability"])
        rating = int(np.clip(round(3.2 + 1.5 * (actual / max(p50, 1) - 0.85) + 0.8 * on_time + self.rng.normal(0, 0.5)), 1, 5))
        self.add("result", participation_id=pid, actual_reach=actual, actual_engagement=int(actual * per_view),
                 conversions=conversions, delivered_on_time=on_time, client_rating=rating,
                 measured_at=iso(end + timedelta(days=7)))

    def _planned_campaigns(self):
        fixed = [("Gruenwerk Foods", "sales", 15000, date(2026, 10, 19), "AT"),
                 ("Nordstride", "awareness", 25000, date(2026, 11, 2), "DE"),
                 ("Kontoro", "app_installs", 20000, date(2026, 11, 16), "AT")]
        extra_brands = [b for b in self.brand_id if b not in {f[0] for f in fixed} and self.brand_cat[self.brand_id[b]] != "alcoholic_beverages"]
        for b in self.rng.choice(extra_brands, 3, replace=False):
            fixed.append((str(b), self.choice(["awareness", "traffic", "sales"]), float(round(self.lognorm(15000, 0.4) / 500) * 500),
                          self.rand_date(date(2026, 11, 20), date(2026, 12, 10)), "AT"))
        self.planned = []
        for k, (brand, goal, budget, start, country) in enumerate(fixed):
            cid, bid, end = self._make_campaign(brand, goal, budget, start, "planned", country)
            self.planned.append((cid, brand, goal, budget, start, end, country))
            cat = self.brand_cat[bid]
            comps = [r["competitor_id"] for r in self.t["brand_competitor"] if r["brand_id"] == bid]
            niches = CATEGORY_NICHES[cat]
            matching = [i for i, x in self.inf.items() if x["status"] == "active" and x["primary"] in niches
                        and "undisclosed_advertising" not in x["flags"]]
            self.rng.shuffle(matching)
            if len(matching) < 3:
                matching += [i for i, x in self.inf.items() if x["status"] == "active" and i not in matching][:3]
            for iid in matching[:2]:                      # competitor promoted in last 90 days
                aid = self.inf[iid]["accounts"][0]
                comp = int(self.choice(comps))
                comp_name = next(n for n, i in self.brand_id.items() if i == comp)
                lo = max(start - timedelta(days=85), TODAY - timedelta(days=70))
                hi = min(start - timedelta(days=10), TODAY - timedelta(days=5))
                d = self.rand_date(lo, hi)          # always inside the 90 days before campaign start
                subj = self.choice(NICHE_SPEC[self.inf[iid]["primary"]][3 if self.inf[iid]["lang"] == "de" else 4])
                tpl = self.choice(DE_SPONSORED if self.inf[iid]["lang"] == "de" else EN_SPONSORED)
                sn = self.snap(aid, d)
                self.add("content_item", account_id=aid, posted_at=iso(d), format="reel" if self.acc[aid]["platform"] == "instagram" else "video",
                         caption=tpl.format(b=comp_name, s=subj), hashtags="#" + NICHE_SPEC[self.inf[iid]["primary"]][5][0] + " #werbung",
                         views=int(sn["views"]), likes=int(sn["er"] / 100 * sn["f"] * 0.7), comments=int(sn["er"] / 100 * sn["f"] * 0.08),
                         is_sponsored=1, brand_id=comp, is_disclosed=1, embedding=None)
                self.truth.append(dict(trap_type="competitor_conflict", entity_type="influencer", entity_id=iid, related_id=cid,
                                       detail=f"promoted competitor {comp_name} on {iso(d)}"))
            if k < 3:                                      # exclusivity contract with a competitor
                iid = matching[2]
                comp = int(self.choice(comps))
                self.add("exclusivity", influencer_id=iid, brand_id=comp, valid_from="2026-07-15", valid_to="2027-02-28")
                self.truth.append(dict(trap_type="exclusivity_block", entity_type="influencer", entity_id=iid, related_id=cid,
                                       detail="exclusive to a competitor 2026-07-15..2027-02-28"))
        for _ in range(4):                                  # unrelated, expired exclusivities (noise)
            iid = int(self.choice(list(self.inf)))
            self.add("exclusivity", influencer_id=iid, brand_id=int(self.choice(list(self.brand_id.values()))),
                     valid_from="2025-03-01", valid_to="2025-12-31")

    # ------------------------------------------------------------ outputs
    def write_briefs(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        tpl = {
            "Gruenwerk Foods": ("Gruenwerk Protein Bowls", "Plant-based ready meals (chilled), 5.99 EUR per bowl, launching in Austrian supermarkets.",
                                "Women and men 22-40 in Austria, urban, interested in healthy eating and cooking without meat.",
                                "friendly, fresh, practical; no preaching about veganism", "alcohol, gambling, other plant-based ready-meal brands"),
            "Nordstride": ("Nordstride Trail Pro running shoe", "Lightweight trail running shoe, 149 EUR, sustainable recycled materials, autumn launch.",
                           "Active runners 25-45 in Germany and Austria, also hikers; mix of beginners and ambitious amateurs.",
                           "authentic, outdoorsy, performance-oriented; honest test reviews preferred", "other sportswear brands, gambling"),
            "Kontoro": ("Kontoro mobile banking app", "Free mobile current account with automatic round-up savings, targeting first-time savers.",
                        "Adults 22-38 in Austria, early career, interested in personal finance and budgeting.",
                        "educational, trustworthy, no get-rich-quick promises", "crypto, gambling, other banking or finance apps"),
        }
        for cid, brand, goal, budget, start, end, country in self.planned[:3]:
            title, desc, aud, tone, restr = tpl[brand]
            text = (f"# Product brief: {title}\n\n"
                    f"**Brand:** {brand} (fictional)  \n**Planned campaign id in database:** {cid}  \n"
                    f"**Goal:** {goal}  \n**Budget:** {budget:,.0f} EUR  \n**Target country:** {country}  \n"
                    f"**Timeline:** {iso(start)} to {iso(end)}\n\n"
                    f"## Product\n{desc}\n\n## Target audience\n{aud}\n\n## Desired tone\n{tone}\n\n"
                    f"## Restrictions\nNo influencers who promote: {restr}.\n")
            (folder / f"brief_{brand.lower().replace(' ', '_')}.md").write_text(text, encoding="utf-8")

    def frames(self) -> dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v) for k, v in self.t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=80, help="number of influencers")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    g = Gen(args.seed, args.n)
    g.build_taxonomy_and_brands()
    g.build_influencers()
    g.build_accounts_and_snapshots()
    g.plant_trapped_accounts()
    g.emit_snapshots_and_audience()
    g.build_rate_cards()
    g.assign_flags()
    g.build_consent_traps()
    g.emit_content()
    g.build_campaigns()

    frames = g.frames()
    csv_dir = args.out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "ground_truth").mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(csv_dir / f"{name}.csv", index=False)
    pd.DataFrame(g.truth).to_csv(args.out / "ground_truth" / "planted_traps.csv", index=False)
    g.write_briefs(args.out / "test_briefs")

    db = args.out / "influencer_db.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    con.execute("PRAGMA foreign_keys = ON")
    con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)",
                    [("as_of", iso(TODAY)), ("seed", str(args.seed)), ("synthetic", "true")])
    for name in ["niche", "influencer", "influencer_niche", "account", "account_snapshot", "audience_snapshot", "brand",
                 "brand_competitor", "content_item", "rate_card", "safety_flag", "exclusivity", "campaign",
                 "participation", "result"]:
        frames[name].to_sql(name, con, if_exists="append", index=False)
    con.commit()
    con.close()

    print(f"seed={args.seed}  influencers={args.n}")
    for name, df in frames.items():
        print(f"  {name:<20}{len(df):>6} rows")
    print(f"  planted traps       {len(g.truth):>6}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
