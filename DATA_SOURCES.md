# Data sources and calibration

Research date: 9 October 2026. This file documents what public data exists for influencer
selection, why it was **not** used directly, and which published benchmarks the synthetic
dataset is calibrated to.

## 1. What public datasets exist

| Source | Content | Why it is not enough |
|---|---|---|
| Hugging Face: *mr-ai-marketer-influencers-public* (vivekchakraverty) | 14,632 Instagram creator profiles: handle, bio, follower/following/post counts, niche, verification flag, last post date | Single snapshot, Instagram only. No rates, audience demographics, engagement history or campaign results. Contact columns removed by the publisher. |
| Kaggle: *Top 200 Instagram Influencers* (referenced by the GitHub project `shauvar/dissertation`) | Top accounts with follower and engagement figures | Celebrity-scale accounts only (mega tier), not representative of an agency's partner pool. |
| Bright Data sample (`luminati-io/Instagram-dataset-samples`) | ~2,180 Instagram profiles of coding influencers with engagement score, categories, hashtags, brand affiliation | Narrow niche, sample of a commercial dataset, no pricing or results. |
| Commercial scrapers and providers (Apify actors, Modash, HypeAuditor, IQFluence) | Profile stats, engagement, audience-quality estimates | Paid or terms-of-service restricted; Instagram and TikTok limit logged-out access. |

**Gaps in every public source:** rate cards, audience demographics over time, campaign
predictions versus results, brand conflicts and exclusivity, delivery reliability. These are
exactly the fields a selection decision needs, and they are normally proprietary to the agency.

**Legal note:** creator profiles are personal data under GDPR. Scraping and storing them
requires a lawful basis, retention limits (see `influencer.consent_until`) and a data-minimisation
concept. This is a further reason the repository ships fully synthetic data.

**Technical note:** these datasets could not be downloaded inside the sandbox where this
repository was prepared (network access was restricted), so no real records were
copied or sampled.

## 2. Published benchmarks used for calibration

All values are in the `CALIBRATION BLOCK` at the top of `generate_dataset.py`.

### Engagement rate by audience size

Sources disagree by a factor of 2-4 depending on method (per follower vs per view, active
creators only, sample). The generator uses mid-range medians per follower.

| Tier | Followers | Reference ranges found | Generator median (Instagram) |
|---|---|---|---|
| Nano | 1K-10K | 2.5-4.0 % (Socialinsider via IQFluence); 6.7 % (Visionary Marketing) | 3.2 % |
| Micro | 10K-100K | 1.3-1.8 %; IQFluence median 0.97 %; 3.9 % (Visionary Marketing) | 1.5 % |
| Mid | 100K-500K | 0.8-1.2 %; 2.4 % (Visionary Marketing) | 1.0 % |
| Macro | 500K-1M | 0.5-0.8 %; 1.4 % (Visionary Marketing) | 0.65 % |
| Mega | 1M+ | 0.3-0.5 %; 0.9 % (Visionary Marketing) | 0.40 % |

TikTok and YouTube medians are set relative to Instagram (TikTok higher, YouTube lower)
following the platform comparisons in the same sources. The generator defines
`engagement_rate` as interactions divided by **followers** for all platforms, so values
are comparable.

The Socialinsider-based ranges are Instagram-specific; the Visionary Marketing figures are
cross-platform tier medians and run higher.

Pattern that is reproduced because every source agrees on it: engagement falls as audience
size grows, and nano/micro creators engage several times more than mega creators.

### Price per sponsored post (DACH, EUR)

| Source | Nano | Micro |
|---|---|---|
| Business Punk, German market 2026 (feed post, 1K-10K / 10K-50K) | 50-250 | 250-1,000 |
| Business Punk (reel) | 100-400 | 400-2,000 |
| Influee 2026 (Instagram, nano post) | 25-150 | |
| Accountable (nano / micro post) | 10-100 | 100-500 |

The generator fits a power law `price = 0.1205 * followers^0.82` for a feed post, scales
reels by 1.8 and story sets by 0.7, applies niche premiums (finance 1.5x, tech 1.2x) and
adjusts for engagement quality. Resulting medians (EUR, validated by the test script):
nano about 70, micro about 550, mid about 1,900, macro about 7,300.

The same sources note that usage rights and exclusivity can double a nano price (for
example 200 EUR becomes 400 EUR with 90-day usage rights and a 30-day exclusivity window).
This is why `rate_card` carries `usage_rights_months` and `exclusivity_premium_pct`.

### Market structure

* Statista and other sources describe the Austrian market as micro-influencer oriented,
  and a German survey puts more than three quarters of influencers in the nano or micro
  tiers. The generator draws 33 % nano, 40 % micro, 15 % mid, 9 % macro, 3 % mega.
* Country split follows a DACH agency's likely pool: 45 % AT, 30 % DE, 10 % CH, 15 % other.

## 3. What is invented (and labelled so)

* All people, handles, bios, captions, cities-to-person pairings.
* All brands (fictional names; category and competitor relations are invented).
* Audience gender/age split per niche, home-country audience share, fake-follower shares,
  posting frequency, growth rates, reach-per-follower factors.
* Campaigns, fees, predictions, results, ratings, delivery behaviour.
* The "naive-v1" predictor is deliberately optimistic by tier (nano -5 % to mega -28 %),
  so the calibration page has a real effect to detect. Historic reach comes out about
  13 % below prediction on average.

## 4. Planted traps (ground truth)

`data/ground_truth/planted_traps.csv` lists deliberately planted problems so the application
can be tested against known answers:

| Trap | Count | How the app should catch it |
|---|---|---|
| `fake_follower_spike` | 5 | followers jump over 50 % in one month, engagement halves, fake share above 30 % |
| `stale_stats` | 5 | latest snapshot older than 90 days |
| `expired_rate_card` | 6 | every rate card of the influencer is past `valid_to` |
| `consent_expired` | 3 | `consent_until` in the past: exclude and flag for deletion |
| `competitor_conflict` | 12 | sponsored post for a competitor in the 90 days before a planned campaign |
| `exclusivity_block` | 3 | exclusivity contract with a competitor overlapping a planned campaign |

Accounts with spiked followers also deliver about 30 % of predicted reach in later
campaigns, versus about 90 % for normal accounts.

## 5. Source list

* Hugging Face dataset page: huggingface.co/datasets/vivekchakraverty/mr-ai-marketer-influencers-public
* GitHub: shauvar/dissertation (Engagement-Based Influencer Marketing Effectiveness)
* GitHub: luminati-io/Instagram-dataset-samples
* IQFluence: Influencer Engagement Rate Benchmarks 2026; Social Media Benchmarks 2026
* Visionary Marketing: Influencer Marketing by Tier Statistics 2026
* Business Punk: So viel verdienen Influencer wirklich (German market, 2026)
* Influee: Instagram Influencer Preise 2026
* Accountable: Influencer income per post (German)
* Statista: Influencer-Werbung Österreich (market outlook)
* CreatorDB, Favikon, Stack Influence: engagement rate formulas and tier benchmarks

Figures from these pages are secondary sources of varying rigour. Treat the dataset as a
plausible simulation, not as market research.
