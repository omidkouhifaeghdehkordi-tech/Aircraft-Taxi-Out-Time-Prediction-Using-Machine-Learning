"""Add hourly Open-Meteo wind data using the first-movement timestamp."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

BCN_LATITUDE = 41.2974
BCN_LONGITUDE = 2.0833
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": BCN_LATITUDE,
            "longitude": BCN_LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "kn",
            "timezone": "UTC",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()["hourly"]
    return pd.DataFrame(
        {
            "weather_hour_utc": pd.to_datetime(payload["time"], utc=True),
            "wind_speed_kt": payload["wind_speed_10m"],
            "wind_direction_deg": payload["wind_direction_10m"],
        }
    )


def enrich_weather(input_path: Path, output_path: Path) -> pd.DataFrame:
    flights = pd.read_csv(input_path)
    required = {
        "first_groundspeed_positive_ts_utc",
        "takeoff_event_utc",
        "takeoff_time_utc",
    }
    missing = sorted(required - set(flights.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    flights["first_groundspeed_positive_ts_utc"] = pd.to_datetime(
        flights["first_groundspeed_positive_ts_utc"], errors="coerce", utc=True
    )
    event_takeoff = pd.to_datetime(flights["takeoff_event_utc"], errors="coerce", utc=True)
    summary_takeoff = pd.to_datetime(flights["takeoff_time_utc"], errors="coerce", utc=True)
    flights["label_takeoff_utc"] = event_takeoff.fillna(summary_takeoff)
    flights["taxi_time_minutes"] = (
        flights["label_takeoff_utc"] - flights["first_groundspeed_positive_ts_utc"]
    ).dt.total_seconds() / 60.0

    # Weather is aligned to the prediction timestamp, not the future takeoff time.
    flights["weather_hour_utc"] = flights["first_groundspeed_positive_ts_utc"].dt.round("h")
    valid_times = flights["weather_hour_utc"].dropna()
    if valid_times.empty:
        raise ValueError("No valid first-movement timestamps were found.")

    weather = fetch_weather(valid_times.min().date().isoformat(), valid_times.max().date().isoformat())
    enriched = flights.merge(weather, on="weather_hour_utc", how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    return enriched


def main() -> None:
    args = parse_args()
    enriched = enrich_weather(args.input, args.output)
    print(f"Saved {len(enriched)} weather-enriched rows to {args.output}")


if __name__ == "__main__":
    main()
