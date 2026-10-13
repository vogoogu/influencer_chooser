# Influencer Chooser: dataset

Synthetic but realistic influencer database for a decision-support tool that matches
influencers to a product brief. It models **evidence over time** (monthly stats, quarterly
audience, rate cards, predictions versus results) rather than a flat list of creators.

> All people, handles, brands and numbers are **fictional**. Distributions are calibrated
> to published 2026 benchmarks, see [DATA_SOURCES.md](../DATA_SOURCES.md). Do not use this
> data for real business decisions.

## Quick start

```bash
pip install -r requirements.txt
python generate_dataset.py            # regenerates everything, deterministic (seed 42)
python tests/validate_dataset.py      # 19 integrity, calibration and trap checks
```

Options: `--seed 7` for another random world, `--n 150` for more influencers.

The repository already contains the generated output, so you can use it without running anything.

## Contents

```
data/
├── influencer_db.sqlite          ready-to-use database (schema.sql + all tables)
├── csv/                          the same tables as CSV (UTF-8)
├── ground_truth/planted_traps.csv  known problems hidden in the data, for testing
└── test_briefs/                  3 product briefs (markdown) for the upload step
schema.sql                        tables, indexes, and views
generate_dataset.py               generator with a calibration block at the top
tests/validate_dataset.py         validation
DATA_SOURCES.md                   research, benchmarks, what is invented
```

## Size (seed 42)

| Table | Rows | Meaning |
|---|---|---|
| `influencer` | 80 | people in DACH plus a few UK/IT/CZ/HU |
| `account` | 117 | Instagram, TikTok, YouTube accounts |
| `account_snapshot` | 2,076 | monthly stats, Apr 2025 to Sep 2026 |
| `audience_snapshot` | 692 | quarterly gender, age, top countries |
| `content_item` | 1,422 | posts with captions (for embeddings), sponsored flag, disclosure |
| `rate_card` | 256 | price per deliverable with usage rights and validity |
| `niche` | 29 | two-level taxonomy (8 parents, 21 leaves) |
| `brand`, `brand_competitor` | 41, 68 | fictional brands and competitor graph |
| `campaign`, `participation`, `result` | 68, 289, 282 | history, 2 running, 6 planned |
| `safety_flag`, `exclusivity` | 16, 7 | brand-safety and contract constraints |
| `run`, `run_item` | 0 | empty, written by the app as an audit trail |

## Data model

```
influencer 1─n account 1─n account_snapshot     (monthly time series)
                      1─n audience_snapshot     (quarterly)
                      1─n content_item
influencer n─m niche (tree)       influencer 1─n rate_card / safety_flag / exclusivity
brand n─m brand (competitors)     brand 1─n campaign 1─n participation 1─1 result
campaign 1─n run 1─n run_item     (written by the application)
```

Conventions:

* `engagement_rate` = interactions / **followers** in percent, for every platform.
* `source` and `confidence` on each snapshot: `api` 0.95, `csv_import` 0.80, `manual` 0.70,
  `ai_estimated` 0.40. Use `confidence` to widen uncertainty in the Monte Carlo step.
* `participation.reach_p10/p50/p90` were predicted **before** the campaign by a deliberately
  optimistic `naive-v1` model. `result.actual_reach` is what happened.
* `embedding` and `profile_embedding` are empty. The app computes them from `bio_text` and
  `caption` (for example with sentence-transformers) and stores them as BLOBs.
* Dates are ISO strings. "Today" in the dataset is **2026-10-09**.

## Built-in views

| View | Purpose |
|---|---|
| `v_latest_snapshot` | latest stats per account |
| `v_influencer_reliability` | on-time rate, average client rating, actual vs predicted reach |
| `v_prediction_error` | per participation ratio and p10-p90 hit, input for the calibration page |
| `v_campaign_conflicts` | competitor promotion in the last 90 days and exclusivity overlaps |

## Example queries

Hard filter: available, consented, no severe safety flag, not conflicted.

```sql
SELECT i.id, i.display_name
FROM influencer i
WHERE i.status = 'active'
  AND i.availability <> 'unavailable'
  AND i.consent_until >= '2026-10-09'
  AND NOT EXISTS (SELECT 1 FROM safety_flag f WHERE f.influencer_id = i.id AND f.severity >= 3)
  AND NOT EXISTS (SELECT 1 FROM v_campaign_conflicts c
                  WHERE c.campaign_id = :campaign_id AND c.influencer_id = i.id);
```

Engagement quality and cost per 1,000 views for Instagram reels:

```sql
SELECT i.display_name, s.followers, s.engagement_rate,
       s.engagement_rate * (1 - s.fake_follower_pct / 100.0) AS quality_adjusted_er,
       r.price_eur * 1000.0 / s.avg_views                    AS cost_per_1000_views
FROM v_latest_snapshot s
JOIN account a    ON a.id = s.account_id AND a.platform = 'instagram'
JOIN influencer i ON i.id = a.influencer_id
JOIN rate_card r  ON r.influencer_id = i.id AND r.platform = 'instagram' AND r.deliverable = 'reel'
ORDER BY quality_adjusted_er DESC;
```

Calibration: how optimistic were the predictions, by deliverable?

```sql
SELECT p.deliverable, COUNT(*) AS n,
       ROUND(AVG(e.ratio), 2)            AS avg_actual_over_predicted,
       ROUND(AVG(e.inside_p10_p90), 2)   AS share_inside_p10_p90
FROM v_prediction_error e JOIN participation p ON p.id = e.participation_id
GROUP BY p.deliverable;
```

## Using it with the Streamlit app

```python
import sqlite3, pandas as pd
con = sqlite3.connect("data/influencer_db.sqlite")
influencers = pd.read_sql("SELECT * FROM influencer", con)
```

Suggested flow to test the app end to end:

1. Upload `data/test_briefs/brief_nordstride.md` (or the other two).
2. Let the AI build the profile, apply hard filters, score, optimise the mix.
3. Compare the output with `data/ground_truth/planted_traps.csv`: the app should avoid or flag
   the influencers listed there for that campaign id (competitor promotion, exclusivity),
   warn about spiked, stale or expired records, and widen reach ranges for
   low-confidence data.

## Replacing synthetic data with real data

Keep `schema.sql` and load real rows into the same tables. The minimum viable set is
`influencer`, `account`, `account_snapshot`, `rate_card`. Everything else makes the tool
smarter but is optional. Check GDPR first: you need a lawful basis, a retention date
(`consent_until`), and you should not store private contact data that was not published by
the creator.

## License

Add a license file of your choice. The synthetic data contains no personal data and can be
shared freely.
