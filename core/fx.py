"""Currency conversion with the free Frankfurter API (ECB rates). Falls back to static rates offline."""
from __future__ import annotations

from . import config

URL = "https://api.frankfurter.dev/v1/latest"


def to_eur(amount: float, currency: str, timeout: float = 4.0) -> tuple[float, str]:
    """Return (amount_in_eur, source) where source is 'frankfurter' or 'static fallback'."""
    currency = currency.upper()
    if currency == "EUR":
        return float(amount), "none needed"
    try:
        import requests
        r = requests.get(URL, params={"base": currency, "symbols": "EUR"}, timeout=timeout)
        r.raise_for_status()
        return float(amount) * float(r.json()["rates"]["EUR"]), "frankfurter"
    except Exception:
        return float(amount) * config.STATIC_FX_TO_EUR.get(currency, 1.0), "static fallback"
