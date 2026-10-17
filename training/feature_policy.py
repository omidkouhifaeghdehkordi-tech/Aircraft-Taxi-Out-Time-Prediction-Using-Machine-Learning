"""Feature availability policy for taxi-time modelling."""

from __future__ import annotations

from collections.abc import Iterable

TIME_COLUMN = "split_time_utc"
TARGET_COLUMN = "taxi_time_minutes"

PUBLIC_FEATURES = [
    "is_weekend",
    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
    "flight_prefix",
    "aircraft_family",
    "aircraft_size",
    "wind_speed_kt",
    "wind_dir_sin",
    "wind_dir_cos",
]

FORBIDDEN_FEATURES = {
    "taxi_time": "direct representation of the target",
    "taxi_time_minutes": "target column",
    "takeoff_event_utc": "known only after takeoff",
    "taxiout_distance_m": "computed from the completed taxi trajectory",
    "taxiout_distance_km": "computed from the completed taxi trajectory",
    "taxiout_track_points_used": "post-processed trajectory information",
    "queue_size_estimated": "uses the target flight's eventual takeoff time",
    "valid_rank_in_day": "may encode final collection or takeoff order",
    "runway_takeoff": "historical actual runway is not available from the public feed at prediction time",
    "runway_side": "derived from the historical actual takeoff runway",
    "runway_heading_deg": "derived from the historical actual takeoff runway",
    "headwind_kt": "depends on the historical actual takeoff runway",
    "crosswind_kt_abs": "depends on the historical actual takeoff runway",
}


def validate_feature_policy(columns: Iterable[str]) -> None:
    columns = set(columns)
    required = set(PUBLIC_FEATURES) | {TIME_COLUMN, TARGET_COLUMN}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    overlap = sorted(set(PUBLIC_FEATURES) & set(FORBIDDEN_FEATURES))
    if overlap:
        raise AssertionError(f"Public feature policy contains forbidden fields: {overlap}")


def forbidden_reasons() -> dict[str, str]:
    return dict(FORBIDDEN_FEATURES)
