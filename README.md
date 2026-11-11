# RR Pressure Index - the method

**Who performs best when the game is on the line?** This project measures how T20 batters and bowlers perform in
high-pressure situations *compared with their own usual level*, and ranks who delivers most consistently.

The value of this repository is the **method**, documented end to end and reproducible. It contains no data and no
results: if you have the SupeRR Selector Season 2 data pack, upload it on the **Run on a data pack** page and the full
pipeline produces every result in a few minutes.

## The method in eight steps
1. **Data reconstruction** - legal balls, credit rules, team totals rebuilt where extras' run values are missing;
   winners re-derived from the chase structure and validated against official results.
2. **Win probability** - per-innings spline logistic model, cross-validated by match, calibration-checked, tested on
   later seasons.
3. **Pressure = leverage** - the expected swing in win probability from each ball; the top 20% are high-pressure balls.
   A transparent rule-based definition is computed alongside for comparison.
4. **Context-fair baselines** - expected runs, dismissal and wicket probability for each ball given phase, wickets,
   required rate, bowler type vs batting hand, scoring era, competition and opposition.
5. **Ratings** - A: pressure quality, B: pressure vs the player's own baseline, C: consistency across matches; all
   shrunk for small samples and combined into the Pressure Reliability Score.
6. **Fair comparison** - associate opposition and A-team cups down-weighted, recency weights, IPL vs T20I checks.
7. **Reliability** - 1,000 bootstrap resamples by match, sensitivity variants and split-half persistence, giving
   Robust / Likely / Sensitive labels.
8. **Transparency** - every formula, setting and evidence table is on the Methodology page.

## Run it
1. Python 3.11+, then: `python -m venv .venv`, activate it, `pip install -r requirements-dev.txt`
2. Set a password: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and edit it
3. `streamlit run app/Home.py`, log in, open **Run on a data pack** and upload the pack (.zip)

Or from the command line: `python scripts/setup_data.py path/to/pack.zip` then
`python -m pressure.run --config config/settings.yaml`. Tests: `python -m pytest`.

All settings are in `config/settings.yaml` (+ `config/settings_overrides.yaml`). The data pack is not included and must
never be committed - see SECURITY.md.
