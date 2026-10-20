"""Central configuration. Nothing here imports Streamlit at module level, so the core
package can be unit-tested without a running app."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_BASELINE = ROOT / "data" / "influencer_db.sqlite"          # committed, read-only baseline
DB_WORKING = Path(os.getenv("INFLUENCER_DB", ROOT / "data" / "working.sqlite"))  # app writes here
BRIEF_DIR = ROOT / "data" / "test_briefs"

# AI goes through OpenRouter (OpenAI-compatible API). The key is OPENROUTER_API_KEY, the model any OpenRouter slug that
# supports tool calling, for example "anthropic/claude-sonnet-5.5" or "openai/gpt-5.2".
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
MODEL_VERSION_TAG = "scoring-v1"        # stored with every saved decision
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")   # "tfidf" or "sentence-transformers"
ST_MODEL = os.getenv("ST_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# ------------------------------------------------------------------ scoring
CRITERIA = ["topic_fit", "audience_match", "engagement_quality", "cost_efficiency",
            "brand_safety", "reliability", "past_performance"]
CRITERIA_LABELS = {
    "topic_fit": "Topic fit",
    "audience_match": "Audience match",
    "engagement_quality": "Engagement quality",
    "cost_efficiency": "Cost efficiency",
    "brand_safety": "Brand safety",
    "reliability": "Reliability",
    "past_performance": "Past performance",
}
CRITERIA_HELP = {
    "topic_fit": "Semantic similarity between brief and the influencer's bio and posts, plus niche overlap.",
    "audience_match": "Share of the audience that matches target gender, age range and countries.",
    "engagement_quality": "Engagement rate adjusted for fake followers, relative to peers of the same platform and size.",
    "cost_efficiency": "Expected views per euro for the chosen deliverable, as a percentile among all accounts.",
    "brand_safety": "Starts at 1 and drops with the severity of active safety flags.",
    "reliability": "Relationship score from the account manager combined with on-time delivery and client ratings.",
    "past_performance": "Client ratings from past campaigns in the same product category (neutral if none).",
}
PRESETS = {
    "Balanced": dict(topic_fit=5, audience_match=5, engagement_quality=4, cost_efficiency=3,
                     brand_safety=4, reliability=3, past_performance=2),
    "Brand fit first": dict(topic_fit=9, audience_match=7, engagement_quality=3, cost_efficiency=1,
                            brand_safety=5, reliability=2, past_performance=2),
    "Budget efficient": dict(topic_fit=3, audience_match=3, engagement_quality=5, cost_efficiency=9,
                             brand_safety=3, reliability=2, past_performance=1),
    "Safe and reliable": dict(topic_fit=3, audience_match=3, engagement_quality=3, cost_efficiency=2,
                              brand_safety=9, reliability=9, past_performance=5),
    "Reach and engagement": dict(topic_fit=2, audience_match=4, engagement_quality=9, cost_efficiency=6,
                                 brand_safety=2, reliability=1, past_performance=1),
}

PRIMARY_DELIVERABLE = {"instagram": "reel", "tiktok": "video", "youtube": "integration"}
DELIVERABLE_REACH_MULT = {"feed_post": 0.7, "story_set": 0.45, "reel": 1.0, "video": 1.0,
                          "integration": 1.0, "dedicated_video": 1.0}
# conversions per reached person by campaign goal (clicks, sales or installs): editable assumptions
GOAL_CONV_RATE = {"awareness": 0.0005, "traffic": 0.012, "sales": 0.0035, "app_installs": 0.004}
GOALS = list(GOAL_CONV_RATE)

FORECAST_METHODS = {"forest": "Random forest (learned from results)", "tier": "Tier correction (simple)", "naive": "Naive (average views)"}

STALE_DAYS = 90
SPIKE_GROWTH = 0.50          # month-over-month follower growth that looks suspicious
FAKE_PCT_WARN = 30.0
LOW_CONFIDENCE = 0.60

PLATFORMS = ["instagram", "tiktok", "youtube"]
FLAG_TYPES = ["alcohol_promotion", "gambling_promotion", "political_content", "controversy",
              "undisclosed_advertising", "crypto_promotion", "profanity"]
TONES = ["casual", "humorous", "educational", "inspirational", "premium", "authentic_raw"]
COUNTRIES = ["AT", "DE", "CH", "IT", "GB", "CZ", "HU"]

STATIC_FX_TO_EUR = {"EUR": 1.0, "CHF": 1.05, "USD": 0.92, "GBP": 1.17}   # fallback if API is offline


def get_secret(name: str) -> str | None:
    """Read a secret from Streamlit secrets or the environment (never from the repo)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)          # raises if no secrets file exists
    except Exception:
        return None
