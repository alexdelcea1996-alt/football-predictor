# Football Match Prediction System

A production-ready machine learning system for predicting football match outcomes (Home Win, Draw, Away Win) with calibrated probability estimates.

**Target metrics:** 52-56% accuracy, RPS below 0.22.

> RPS here is the standard definition (sum of the first K-1 cumulative terms,
> divided by K-1). Earlier versions of this project divided by K and included
> the final always-zero term, which understated the score by a third and made
> it incomparable with published numbers.

## Features

- **Ensemble with a fitted blend**: CatBoost + XGBoost + logistic regression,
  plus the Dixon-Coles goal model and the market's own prices as members.
  Weights are fitted to minimize RPS on out-of-fold predictions, so a weak
  member cannot drag the ensemble below its best one.
- **Bookmaker odds**: de-vigged with Shin's method, the strongest publicly
  available signal for match outcomes.
- **Dixon-Coles goal model**: bivariate Poisson with time decay, both as a
  feature source and as an ensemble member. Classifiers systematically
  under-predict draws; a scoreline distribution does not.
- **Feature engineering**: Elo ratings, rolling form, head-to-head, contextual
  and experience features, with degenerate (all-empty) columns dropped.
- **Honest probability calibration**: fitted on out-of-fold predictions, saved
  and loaded with the model.
- **Temporal cross-validation**: expanding windows that never look forward.
- **Benchmark harness**: compares pipeline configurations on identical folds.
- **CLI interface** and **SHAP explainability**.

## Installation

```bash
# Clone and install
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Get data

Real matches with bookmaker odds (free, no API key, many seasons):

```bash
python scripts/fetch_odds_data.py --seasons 2018-2024
```

Or simulate a league to try the pipeline without downloading anything. The
simulated teams have latent strengths, so the data carries learnable signal,
but its scores say nothing about real-world accuracy:

```bash
football-predictor generate-sample --output data/matches.csv --seasons 3
```

### 2. Train

```bash
football-predictor train data/matches_with_odds.csv --output models/
```

The output reports each member's score on the holdout next to its blend
weight, so it is visible at a glance when a member is not earning its place.

### 3. Make predictions

```bash
football-predictor predict fixtures.csv --model models/
```

Fixture rows may carry `odds_home`, `odds_draw` and `odds_away`; when they do,
the market features are filled in and the prediction improves accordingly.

### 4. Evaluate with cross-validation

```bash
football-predictor evaluate data/matches_with_odds.csv --folds 5
```

### 5. Compare pipeline configurations

```bash
python scripts/benchmark.py --data data/matches_with_odds.csv
```

Runs every configuration on identical temporal folds, from the pre-audit
behaviour to the full pipeline, alongside reference rows for the bookmakers'
prices and for simply predicting the base rates.

### 6. Tune hyperparameters

```bash
python scripts/optimize_hyperparams.py --data data/matches_with_odds.csv --trials 50
```

Trials are scored on temporal CV folds inside the training block; the final
holdout is scored once, afterwards. Results land in `models/best_params.json`
and are picked up automatically by the next training run.

> The `models/best_params.json` committed to this repository predates the fix:
> it was tuned with the holdout as the objective *and* the early-stopping set,
> so those values are fitted to that particular block. Re-run the tuner on
> your data before relying on them.

## Programmatic Usage

```python
from football_predictor.training.trainer import ModelTrainer
from football_predictor.prediction.predictor import MatchPredictor
import pandas as pd

# Train
df = pd.read_csv("data/matches.csv")
trainer = ModelTrainer(output_dir="models")
results = trainer.train(df, test_size=0.2)
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.2%}")
print(f"Test RPS: {results['test_metrics']['rps']:.4f}")

# Predict
predictor = MatchPredictor()
predictor.load("models")

predictions = predictor.predict_fixtures([
    {"date": "2024-01-15", "home_team": "Arsenal", "away_team": "Liverpool"},
    {"date": "2024-01-16", "home_team": "Man City", "away_team": "Chelsea"},
])

for p in predictions:
    print(f"{p['home_team']} vs {p['away_team']}: {p['predicted_outcome']}")
    print(f"  H: {p['home_win_prob']:.0%}, D: {p['draw_prob']:.0%}, A: {p['away_win_prob']:.0%}")
```

## Data Format

### Input CSV (Historical Matches)

```csv
date,home_team,away_team,home_goals,away_goals,league,season
2024-01-01,Arsenal,Liverpool,2,1,premier_league,2023/24
```

**Required columns:** `date`, `home_team`, `away_team`, `home_goals`, `away_goals`

**Optional columns:** `league`, `season`, `odds_home`, `odds_draw`, `odds_away`,
`home_xg`, `away_xg`, `home_shots`, `away_shots`, `home_possession`,
`away_possession`

Odds columns are decimal prices (2.10, not +110). `scripts/fetch_odds_data.py`
writes them in this format. Columns that turn out to be empty for the whole
dataset, such as xG when the source has none, are dropped rather than being
filled with zeros and fed to the models as noise.

### Output CSV (Predictions)

```csv
date,home_team,away_team,predicted_outcome,home_win_prob,draw_prob,away_win_prob,confidence
```

## Configuration

Set via environment variables or `.env` file:

```env
# API Keys (optional)
FP_API_SPORTMONKS_API_KEY=your_key_here
FP_API_API_FOOTBALL_KEY=your_key_here

# Feature Engineering
FP_FEATURE_ROLLING_WINDOWS=[5,10,20]
FP_FEATURE_ELO_K_FACTOR=20
FP_FEATURE_DECAY_FACTOR=0.95

# Model Training
FP_MODEL_CV_FOLDS=5
FP_MODEL_EARLY_STOPPING_ROUNDS=50

# Data sources (only needed for the football-data.org scripts)
FOOTBALL_DATA_API_KEY=your_key_here
```

## Project Structure

```
football_predictor/
├── data/           # API clients, football-data.co.uk loader, simulator
├── features/       # Elo, rolling stats, H2H, contextual, odds, Dixon-Coles
├── models/         # CatBoost, XGBoost, LogReg, Dixon-Coles, blending,
│                   # calibration, tuned-parameter loading, ensemble
├── training/       # CV strategy, hyperopt, trainer
├── evaluation/     # Metrics (RPS, calibration), reporting
├── prediction/     # Prediction interface
├── explainability/ # SHAP explainer
├── cache/          # Feature store
└── cli.py          # Command-line interface

scripts/
├── fetch_odds_data.py       # Historical matches with odds
├── benchmark.py             # Compare pipeline configurations
├── optimize_hyperparams.py  # Optuna tuning on temporal CV
└── predict_today.py         # Predict today's fixtures
```

## What the changes are worth

`scripts/benchmark.py` on four simulated seasons (1,520 matches, three
temporal folds). These numbers validate the pipeline, **not** real-world
accuracy: the simulated market is close to the truth by construction, so treat
the reference rows as an upper bound rather than a forecast of what you will
achieve. Re-run it on your own downloaded data.

| Configuration | Accuracy | RPS | Draw F1 | |
|---|---|---|---|---|
| `legacy` | 53.0% | 0.2005 | 0.017 | pre-audit behaviour |
| `fixed` | 52.4% | 0.2038 | 0.017 | fitted blend + calibration |
| `poisson_features` | 50.9% | 0.2058 | 0.017 | Dixon-Coles as features |
| `poisson_member` | 53.8% | 0.1973 | 0.000 | Dixon-Coles as a member |
| `odds_features` | 52.6% | 0.2002 | 0.065 | odds as features |
| `odds_member` | 55.7% | 0.1922 | 0.043 | odds as a member |
| **`full`** | **55.6%** | **0.1921** | 0.051 | odds + Dixon-Coles as members |
| `full+stacking` | 54.5% | 0.1933 | 0.072 | meta-learner blend |
| *reference: market* | 53.8% | 0.1895 | 0.050 | the bookmakers' own prices |
| *reference: dixon_coles* | 53.7% | 0.1950 | 0.050 | goal model alone |
| *reference: class_prior* | 47.4% | 0.2266 | 0.000 | predicting base rates |

Two things stand out. Handing a probability source to the blender as a member
beats feeding it to the classifiers as features, for both the market
(0.1922 vs 0.2002) and the goal model (0.1973 vs 0.2058) - as features they
are two columns among eighty, as members they are weighted on their merits.
The fitted weights reflect that: the market member takes about 0.58 of the
full blend and the goal model 0.07, with the rest going to XGBoost and the
logistic model.

## How the ensemble is trained

1. Features are computed match by match in chronological order, so a match is
   only ever described by earlier ones.
2. The data is split chronologically; the test block is scored once, at the
   end, and is never used for early stopping, calibration or blending.
3. Members are trained on the training block. Temporal folds inside that
   block produce out-of-fold predictions, which are used to fit each member's
   calibration map and the blend weights. Fitting the blend on a single small
   validation block made the weights swing wildly; out-of-fold predictions
   give several times as much data to fit on.
4. Blend weights minimize RPS over the simplex, so they can always fall back
   on "everything to the best member". With `--blend stack` a logistic
   meta-learner is tried as well and kept only when it scores better on the
   same out-of-fold predictions.

## Running Tests

```bash
pytest tests/ -v
```

No test needs network access: data-dependent tests run against the simulator
or small inline fixtures.

## License

MIT
