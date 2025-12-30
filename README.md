# Football Match Prediction System

A production-ready machine learning system for predicting football match outcomes (Home Win, Draw, Away Win) with calibrated probability estimates.

**Target Metrics:**

- Accuracy: 52-56%
- RPS (Ranked Probability Score): < 0.22

## Features

- **Ensemble ML Models**: CatBoost + XGBoost + Logistic Regression with soft voting
- **Comprehensive Feature Engineering**: Elo ratings, rolling stats, head-to-head, contextual features
- **Probability Calibration**: Well-calibrated probability outputs
- **Temporal Cross-Validation**: Prevents data leakage with expanding window CV
- **API Integration**: Supports Sportmonks, API-Football, and CSV data
- **CLI Interface**: Easy-to-use command-line tools
- **SHAP Explainability**: Feature importance and prediction explanations

## Installation

```bash
# Clone and install
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate Sample Data

```bash
football-predictor generate-sample --output data/matches.csv --matches 500
```

### 2. Train Model

```bash
football-predictor train data/matches.csv --output models/
```

### 3. Make Predictions

```bash
football-predictor predict fixtures.csv --model models/
```

### 4. Evaluate with Cross-Validation

```bash
football-predictor evaluate data/matches.csv --folds 5
```

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

**Optional columns:** `league`, `season`, `home_xg`, `away_xg`, `home_shots`, `away_shots`, `home_possession`, `away_possession`

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
```

## Project Structure

```
football_predictor/
├── data/           # API clients, preprocessing, validation
├── features/       # Elo, rolling stats, H2H, contextual features
├── models/         # CatBoost, XGBoost, LogReg, ensemble
├── training/       # CV strategy, hyperopt, trainer
├── evaluation/     # Metrics (RPS, calibration), reporting
├── prediction/     # Prediction interface
├── explainability/ # SHAP explainer
├── cache/          # Feature store
└── cli.py          # Command-line interface
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
