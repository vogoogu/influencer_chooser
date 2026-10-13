# Influencer Chooser

A decision-support tool for a marketing agency. Upload a product brief, and the app filters, scores and
ranks your partner influencers, builds the best campaign mix for a budget, shows the range of likely
outcomes, and learns from real campaign results.

**Principle: AI reads and explains, code decides.** Claude extracts the brief, researches the market,
explains rankings and plays devil's advocate. Every score, ranking, budget split and forecast is calculated
by deterministic Python from the database, so results are repeatable and auditable. Without an API key the
app still works in rule-based mode.

> The bundled data is **synthetic** (fictional people and brands, calibrated to published 2026 benchmarks).
> See [DATA_SOURCES.md](DATA_SOURCES.md) and [docs/DATA.md](docs/DATA.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: add your Anthropic key to enable the AI features. Either paste it into the sidebar (kept for the
browser session only), or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.

First steps in the app: **Product brief** → pick a sample brief → *Load sample brief* → *Analyze brief* →
**Shortlist** → **Campaign mix** → *Save decision*.

## Deploy free on Streamlit Community Cloud

1. Push this repository to GitHub (the folder `data/` with the SQLite file must be included).
2. On share.streamlit.io choose the repo and `app.py` as the main file.
3. Under *Advanced settings → Secrets* paste `ANTHROPIC_API_KEY = "sk-ant-..."` (optional).
4. Deploy. The app copies `data/influencer_db.sqlite` to `data/working.sqlite` on first start.
   Cloud storage is ephemeral: saved decisions and logged results disappear when the app restarts.
   For permanent storage point `INFLUENCER_DB` at a persistent volume or move to Postgres.

## How it works

```
brief (pdf/docx/text) ──► AI profile (or rules) ──► hard filters ──► 7 weighted criteria ──► ranking
                                                         │                                      │
                                              reasons for every exclusion          stability + weight sensitivity
                                                                                                 ▼
 decision log ◄── memo / PDF ◄── risk review ◄── Monte Carlo ◄── optimal mix (integer programming)
      │
      └─► convert to campaign ─► log real results ─► calibration ─► better forecasts next time
```

| Page | What it does |
|---|---|
| Product brief | Reads PDF, Word, text or markdown. Builds an editable profile: category, goal, budget, countries, niches, tone, audience, restrictions. Optional web research. |
| Shortlist | Hard filters (availability, GDPR consent, country, budget share, safety flags, brand conflicts). Scores seven criteria, sliders re-rank live, top-5 stability, weight sensitivity, AI explanations. |
| Campaign mix | Maximises qualified reach, conversions or fit score under budget, creator count, concentration and platform constraints. Compares with "top scores until the budget is gone". Monte Carlo ranges, tornado chart, risk review, memo, PDF/HTML/Markdown/CSV export, save decision. |
| Explorer | Evidence behind the scores: follower and engagement trends, audience, rate cards, flags, content, campaign history. |
| Decision log | Saved decisions with weights, brief and scores (audit trail). Convert to a campaign, log real results. A demo helper simulates outcomes. |
| Calibration | Predicted vs actual reach, bias and interval coverage by audience size, the learned correction, and which known-in-advance features predict success. |

### The seven criteria (all 0 to 1)

| Criterion | Meaning |
|---|---|
| Topic fit | Text similarity between brief and the creator's bio and posts (percentile), plus niche overlap |
| Audience match | Share of the audience in the target gender, age range and countries |
| Engagement quality | Engagement rate adjusted for fake followers, relative to peers of the same platform and size |
| Cost efficiency | Expected views per euro for the chosen format, as a percentile |
| Brand safety | Falls with the severity of active safety flags |
| Reliability | Relationship score blended with on-time delivery and client ratings |
| Past performance | Client ratings in the same product category (neutral without history) |

### Forecasts and the learning loop

Reach is predicted from average views and the format, then corrected by a per-audience-size bias learned
from logged results (shrunk towards the global average when a tier has few results). The spread learned from
the same history, widened for low-confidence data, drives the p10/p90 range and the Monte Carlo simulation.
Log more results and the next forecast improves.

## AI usage and privacy

| Task | Done by |
|---|---|
| Extract the brief into a profile (forced tool call with enum-validated schema) | Claude, fallback: keyword rules |
| Market research | Claude with web search, optional |
| Explain ranked creators, review the mix, draft the memo | Claude, fallback: templates |
| Filters, scores, rankings, optimisation, simulation, calibration | Python only |

Claude receives only facts from the database and the brief, wrapped in tags and marked as data, not
instructions. Turn off *Use AI* in the sidebar to keep everything local. Check GDPR before loading real
creator data: lawful basis, retention (`consent_until`), no private contact data.

## Configuration

| Setting | How |
|---|---|
| API key | sidebar, `ANTHROPIC_API_KEY` env var or Streamlit secret |
| Model | `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`) |
| Database | `INFLUENCER_DB` (default `data/working.sqlite`) |
| Topic similarity | default TF-IDF; `EMBEDDING_BACKEND=sentence-transformers` after `pip install -r requirements-optional.txt` |
| Conversion rates per goal, staleness and spike thresholds | `core/config.py` |

## Repository

```
app.py                 home page
pages/                 Product brief, Shortlist, Campaign mix, Explorer, Decision log, Calibration
core/                  engine (no Streamlit needed): scoring, optimizer, simulation, predict, ai, report, db, ...
data/                  influencer_db.sqlite, csv/, test_briefs/, ground_truth/planted_traps.csv
schema.sql             database schema and views
generate_dataset.py    deterministic synthetic data generator
tests/                 50 tests (core, mocked AI, database, headless pages, dataset validation)
```

Run the tests: `pip install -r requirements-dev.txt && python -m pytest tests -q`
and the dataset checks: `python tests/validate_dataset.py`.

## Known limitations

* AI calls were tested with a fake client. Verify them once with a real key (profile extraction, explanations, web research).
* TF-IDF similarity is weaker than a multilingual embedding model; German and English are bridged through a niche keyword list.
* Conversion rates per goal are assumptions, not measurements. Replace them with your own.
* Audience overlap between creators is not modelled; the risk review only warns about same-niche concentration.
* SQLite on Streamlit Cloud is not persistent.
* Synthetic data: results show the method, not market truth.
