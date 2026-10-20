# Influencer Chooser

A decision-support tool for a marketing agency. Upload a product brief, and the app filters, scores and
ranks your partner influencers, builds the best campaign mix for a budget, shows the range of likely
outcomes, and learns from real campaign results.

**Principle: AI reads and explains, code decides.** An AI model (through OpenRouter) extracts the brief, researches the market,
explains rankings and plays devil's advocate. Every score, ranking, budget split and forecast is calculated
by deterministic Python from the database, so results are repeatable and auditable. Without an API key the
app still works in rule-based mode.

> The bundled data is **synthetic** (fictional people and brands, calibrated to published 2026 benchmarks).
> See [DATA_SOURCES.md](DATA_SOURCES.md) and [docs/DATA.md](docs/DATA.md).

![Shortlist](docs/screenshots/shortlist.jpg)

Design system, layout patterns and responsive behaviour: [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: add your OpenRouter key to enable the AI features. Either paste it into the header Menu (kept for the
browser session only), or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.

First steps in the app: **Product brief** → pick a sample brief → *Load sample brief* → *Analyze brief* →
**Shortlist** → **Campaign mix** → *Save decision*.

To test the upload path, upload `data/test_briefs/Product_Sheet_Kleinstern_Sleepy_Stars.pdf` and select the client brand
*Kleinstern* (enables the competitor check).

## Deploy free on Streamlit Community Cloud

1. Push this repository to GitHub (the folder `data/` with the SQLite file must be included).
2. On share.streamlit.io choose the repo and `app.py` as the main file.
3. Under *Advanced settings → Secrets* paste `OPENROUTER_API_KEY = "sk-or-v1-..."` (optional, see "AI through OpenRouter" below).
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
| Campaign mix | Maximises qualified reach (reach x audience match x topic fit), conversions or fit score under budget, creator count, concentration and platform constraints. Compares with "top scores until the budget is gone". Monte Carlo ranges, tornado chart, risk review, memo, PDF/HTML/Markdown/CSV export, save decision. |
| Explorer | Evidence behind the scores: follower and engagement trends, audience, rate cards, flags, content, campaign history. |
| Decision log | Saved decisions with weights, brief and scores (audit trail). Convert to a campaign, log real results. A demo helper simulates outcomes. |
| Calibration | Predicted vs actual reach, bias and interval coverage by audience size, the learned correction, and which known-in-advance features predict success. |
| Forecast model | Explains the **random forest** step by step: the data, one decision tree, the forest voting, an honest cross-validated test against simple baselines, feature importance, a per-creator explanation and a what-if tool. |

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

### Forecasts: three methods, one honest test

The naive forecast (a creator's average views) is optimistic. The Shortlist page lets you choose how it is corrected:

| Method | What it does |
|---|---|
| Naive | Average views times the format factor, no correction. |
| Tier correction | Multiplies by the average bias found per audience size (shrunk when a tier has few results). |
| **Random forest** (default) | Two scikit-learn forests learned from every logged result: one predicts reach relative to the naive forecast, one predicts conversions per 1,000 reached. Inputs are facts known before the campaign (audience size, engagement vs peers, fake followers, recent follower jumps, rate, relationship, niche fit, the creator's earlier results). |

The forest is judged by 5-fold cross-validation grouped by campaign (no campaign is ever split between training and testing)
against simple baselines, and is only used where it cuts the error by at least 3%. Otherwise the app falls back and says so.
Ranges come from out-of-bag residuals, are widened for low-confidence data, and feed the optimizer and the Monte Carlo simulation.
Training is deterministic (fixed seed), takes about 7 seconds, and repeats automatically when new results are logged.
On the bundled data the reach forest has about 21% lower error than the tier correction and its 80% ranges contain 81% of outcomes.
The **Forecast model** page shows all of this interactively.

## AI through OpenRouter

All AI features use [OpenRouter](https://openrouter.ai) (an OpenAI-compatible API in front of many models). Without a key the app runs in
rule-based mode and everything still works.

1. Create a key at openrouter.ai/keys and add credits.
2. Streamlit Cloud: *App settings → Secrets*, then paste

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   # optional: any OpenRouter model slug that supports tool calling (openrouter.ai/models)
   OPENROUTER_MODEL = "anthropic/claude-sonnet-5.5"
   ```

   Locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, or set the same names as environment variables.
3. In the app, the header shows "AI active" and the Menu shows the model in use. With no secret set, the Menu offers key and model fields
   (kept for the browser session only).

How it calls the API: `POST https://openrouter.ai/api/v1/chat/completions` with a forced function call for structured answers
(brief profile, explanations, AI review) and `provider.require_parameters` so only providers that support tools are used. The optional market
research adds OpenRouter's `openrouter:web_search` server tool (billed by OpenRouter per search). If a model returns JSON as text instead of
a tool call it is still accepted. Failures (bad key, no credits, unknown model, rate limits) are shown in plain words and the app falls back to
rule-based text.

Some models have no provider that accepts a *forced* function choice (OpenRouter then answers 404 "No endpoints found that support the provided
'tool_choice' value"). The client therefore steps down automatically and remembers what worked for the model: forced function call, then
`tool_choice: "required"`, then `"auto"` with an instruction in the prompt, then plain JSON that is validated and parsed. The header Menu shows
which mode is in use. If no mode works, the error names the model, and the app falls back to rule-based text.

## AI usage and privacy

| Task | Done by |
|---|---|
| Extract the brief into a profile (forced tool call with enum-validated schema) | AI model, fallback: keyword rules |
| Market research | AI model with OpenRouter's web search tool, optional |
| Explain ranked creators, review the mix, draft the memo | AI model, fallback: templates |
| Filters, scores, rankings, optimisation, simulation, calibration, **random forest forecasts** | Python only (the AI never produces numbers) |

The model receives only facts from the database and the brief, wrapped in tags and marked as data, not
instructions. Turn off *Use AI* in the header Menu to keep everything local. Check GDPR before loading real
creator data: lawful basis, retention (`consent_until`), no private contact data.

## Configuration

| Setting | How |
|---|---|
| API key | Streamlit secret or env var `OPENROUTER_API_KEY`, or the header Menu field (session only) |
| Model | `OPENROUTER_MODEL` secret or env var, or the header Menu field (default `anthropic/claude-sonnet-5.5`) |
| Database | `INFLUENCER_DB` (default `data/working.sqlite`) |
| Topic similarity | default TF-IDF; `EMBEDDING_BACKEND=sentence-transformers` after `pip install -r requirements-optional.txt` |
| Assumed conversion rates per goal, staleness and spike thresholds | `core/config.py` |
| Forest settings (trees, leaf size, seed, required gain) | `core/forest.py` |

## Repository

```
app.py                 router: grouped navigation (Workflow / Insights), logo, page config
pages/                 Home, Product brief, Shortlist, Campaign mix, Explorer, Decision log, Calibration, Forecast model
core/                  engine (no Streamlit needed): scoring, optimizer, simulation, predict, forest, ai, report, db, ...
                       design system: theme.py (tokens and CSS), components.py (header, stepper, chips, empty states)
static/                bundled Inter font (OFL), logo and icon
data/                  influencer_db.sqlite, csv/, test_briefs/, ground_truth/planted_traps.csv
schema.sql             database schema and views
generate_dataset.py    deterministic synthetic data generator
tests/                 73 tests (core, forest, mocked AI, database, design system, headless pages, dataset validation)
```

Run the tests: `pip install -r requirements-dev.txt && python -m pytest tests -q`
and the dataset checks: `python tests/validate_dataset.py`.

## Known limitations

* The OpenRouter integration is tested against a fake endpoint (request format, forced tool call, secrets handling, errors and retries). Verify it once with your real key: profile extraction, explanations and the optional web research.
* TF-IDF similarity is weaker than a multilingual embedding model; German and English are bridged through a niche keyword list.
* Conversion rates per goal are assumptions, not measurements. Replace them with your own.
* Audience overlap between creators is not modelled; the risk review only warns about same-niche concentration.
* SQLite on Streamlit Cloud is not persistent.
* Synthetic data: results show the method, not market truth.
