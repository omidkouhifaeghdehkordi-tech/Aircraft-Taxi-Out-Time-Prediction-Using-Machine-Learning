"""Build the leakage-safe modelling table from weather-enriched flight rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

FAMILY_BY_TYPE = {
    "A20N": "A320_family",
    "A21N": "A320_family",
    "A319": "A320_family",
    "A320": "A320_family",
    "A321": "A320_family",
    "A332": "A330_family",
    "A333": "A330_family",
    "AT72": "ATR_family",
    "AT73": "ATR_family",
    "AT75": "ATR_family",
    "B38M": "B737_family",
    "B734": "B737_family",
    "B738": "B737_family",
    "B739": "B737_family",
    "B752": "B757_family",
    "B788": "B787_family",
    "B789": "B787_family",
    "CRJX": "CRJ_family",
    "E170": "Embraer_family",
    "E175": "Embraer_family",
    "E190": "Embraer_family",
    "E195": "Embraer_family",
}

HEAVY_TYPES = {"A332", "A333", "A339", "A343", "A346", "A359", "A35K", "B744", "B748", "B772", "B773", "B77W", "B788", "B789"}
LIGHT_TYPES = {"C25A", "C25B", "C25C", "C510", "C525", "C550", "C56X", "LJ35"}

MODEL_COLUMNS = [
    "split_time_utc",
    "taxi_time_minutes",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def flight_prefix(value: object) -> str:
    match = re.match(r"([A-Za-z]+)", str(value or "").strip())
    return match.group(1).upper() if match else "UNKNOWN"


def aircraft_family(value: object) -> str:
    aircraft_type = str(value or "UNKNOWN").strip().upper()
    return FAMILY_BY_TYPE.get(aircraft_type, aircraft_type or "UNKNOWN")


def aircraft_size(value: object) -> str:
    aircraft_type = str(value or "").strip().upper()
    if aircraft_type in HEAVY_TYPES:
        return "Heavy"
    if aircraft_type in LIGHT_TYPES:
        return "Light"
    return "Medium"


def build_model_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "first_groundspeed_positive_ts_utc",
        "taxi_time_minutes",
        "flight_number",
        "callsign",
        "aircraft_type",
        "wind_speed_kt",
        "wind_direction_deg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    output = pd.DataFrame()
    output["split_time_utc"] = pd.to_datetime(
        frame["first_groundspeed_positive_ts_utc"], errors="coerce", utc=True
    )
    output["taxi_time_minutes"] = pd.to_numeric(frame["taxi_time_minutes"], errors="coerce")

    minute_of_day = output["split_time_utc"].dt.hour * 60 + output["split_time_utc"].dt.minute
    day_of_week = output["split_time_utc"].dt.dayofweek
    output["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    output["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    output["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    output["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    output["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    identity = frame["flight_number"].fillna("").where(
        frame["flight_number"].fillna("").str.strip().ne(""), frame["callsign"]
    )
    output["flight_prefix"] = identity.map(flight_prefix)
    output["aircraft_family"] = frame["aircraft_type"].map(aircraft_family)
    output["aircraft_size"] = frame["aircraft_type"].map(aircraft_size)
    output["wind_speed_kt"] = pd.to_numeric(frame["wind_speed_kt"], errors="coerce")
    direction = pd.to_numeric(frame["wind_direction_deg"], errors="coerce")
    output["wind_dir_sin"] = np.sin(np.deg2rad(direction))
    output["wind_dir_cos"] = np.cos(np.deg2rad(direction))

    output = output[MODEL_COLUMNS]
    output = output.dropna().copy()
    output = output[output["taxi_time_minutes"].between(1.0, 60.0)]
    output = output.sort_values("split_time_utc").drop_duplicates().reset_index(drop=True)
    return output


def main() -> None:
    args = parse_args()
    input_frame = pd.read_csv(args.input)
    output_frame = build_model_table(input_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False)
    print(f"Saved {len(output_frame)} model-ready rows to {args.output}")


if __name__ == "__main__":
    main()
