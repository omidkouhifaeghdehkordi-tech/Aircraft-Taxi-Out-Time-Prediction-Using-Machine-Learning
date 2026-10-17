# Reproducibility and leakage audit

## Critical publication issue

Two supplied notebooks contained live-looking Flightradar24 API credentials, including one credential stored in notebook output. They are excluded from the public repository and scrubbed in the separate sanitized archive. Both original credentials should still be revoked or rotated because removing them from files does not invalidate them.

## Feature availability findings

The intended prediction time is the first recorded aircraft movement. The following original fields are not used by the public pipeline:

| Field | Reason for exclusion |
|---|---|
| `takeoff_event_utc`, `taxi_time`, `taxi_time_minutes` | Directly determine or contain the target |
| `taxiout_distance_*`, `taxiout_track_points_used` | Require the completed trajectory |
| `queue_size_estimated` | Collector uses the target flight's eventual takeoff time when counting aircraft ahead |
| `runway_takeoff` and runway-derived wind fields | Historical actual runway is not reliably available from the public feed at prediction time |
| `valid_rank_in_day` | May encode final collection or takeoff order |

## Evaluation finding

The original feature-selection notebook used the newest 20% of rows for permutation importance. A later modelling cell reused the newest 20% as its final test set. That means the later score is not an untouched final-test estimate.

The cleaned public workflow therefore uses forward-chaining cross-validation and does not claim an independent holdout score. A genuinely untouched future collection period should be reserved for the next final evaluation.

The original weather notebook also aligned hourly weather to takeoff time. The new enrichment module aligns weather to the first-movement timestamp so it does not use a future timestamp when constructing model inputs.

## Data-lineage gap

The original archive contained many intermediate CSVs and duplicate files but no single deterministic raw-to-model builder. The refactored project now provides separate flight collection, weather enrichment and model-table preparation modules. The new builder intentionally produces only the deployment-aligned feature set.

## Authorship and redistribution

The submitted report is co-authored by Costin Boiciuc and Omid Kouhifaeghdehkordi. Confirm co-author permission and the course/provider redistribution rules before publishing the full report, original notebooks or complete dataset.
