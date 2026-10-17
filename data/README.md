# Data availability

The supplied research dataset contains 477 rows from April-May 2026. It is not included publicly until the Flightradar24 subscription terms and the team's redistribution permission have been confirmed.

The project separates its data stages:

```text
data/private/fr24_departures.csv       raw retrospective flight records
data/private/departures_weather.csv    first-movement weather enrichment and target
data/private/model_table.csv           leakage-safe training table
```

These files are ignored by Git. Only the model table is accepted by the training module. Its schema is defined in `training/feature_policy.py`.

Never commit API tokens, session cookies or paid-provider credentials. The collector reads the credential only from `FR24_API_TOKEN`.
