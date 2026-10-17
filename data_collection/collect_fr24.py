"""Collect retrospective Barcelona departure records from Flightradar24.

The API credential is read only from the ``FR24_API_TOKEN`` environment
variable. It is never written to CSV, logs, notebooks, or source code.

This collector creates raw research data, including takeoff timestamps needed
to construct the supervised-learning target. The modelling pipeline does not
use takeoff timestamps or the actual takeoff runway as input features.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fr24sdk.client import Client

MADRID_TZ = ZoneInfo("Europe/Madrid")
AIRPORT_FILTER = "outbound:LEBL"

OUTPUT_COLUMNS = [
    "requested_day_local",
    "flight_number",
    "callsign",
    "aircraft_type",
    "registration",
    "origin_iata",
    "origin_icao",
    "destination_iata",
    "destination_icao",
    "runway_takeoff",
    "first_groundspeed_zero_ts_utc",
    "first_groundspeed_positive_ts_utc",
    "first_groundspeed_positive_kt",
    "first_groundspeed_positive_alt_ft",
    "gate_departure_event_utc",
    "takeoff_event_utc",
    "takeoff_time_utc",
    "fr24_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/private/fr24_departures.csv"))
    parser.add_argument("--target-per-day", type=int, default=20)
    parser.add_argument("--summary-limit", type=int, default=1500)
    parser.add_argument("--minimum-call-interval", type=float, default=6.5)
    return parser.parse_args()


def parse_timestamp(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def utc_string(value: object) -> str:
    parsed = parse_timestamp(value)
    return parsed.astimezone(timezone.utc).isoformat() if parsed else ""


def local_day_bounds(day_local: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day_local, dt_time.min, tzinfo=MADRID_TZ)
    end = datetime.combine(day_local + timedelta(days=1), dt_time.min, tzinfo=MADRID_TZ)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


class RateLimiter:
    def __init__(self, minimum_interval: float, max_retries: int = 8):
        self.minimum_interval = minimum_interval
        self.max_retries = max_retries
        self.last_call = 0.0

    def call(self, function, description: str):
        for attempt in range(self.max_retries + 1):
            elapsed = time.time() - self.last_call
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                self.last_call = time.time()
                return function()
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "too many requests" in str(exc).lower()
                if not is_rate_limit or attempt == self.max_retries:
                    raise
                wait_seconds = 60 * (attempt + 1)
                print(f"Rate limit during {description}; retrying in {wait_seconds}s")
                time.sleep(wait_seconds)


def first_ground_movement(track_points: list[object]) -> dict[str, object] | None:
    points = sorted(
        track_points,
        key=lambda point: parse_timestamp(getattr(point, "timestamp", ""))
        or datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    for previous, current in zip(points, points[1:]):
        previous_speed = getattr(previous, "gspeed", None)
        current_speed = getattr(current, "gspeed", None)
        previous_altitude = getattr(previous, "alt", None)
        current_altitude = getattr(current, "alt", None)
        values = [previous_speed, current_speed, previous_altitude, current_altitude]
        if any(value is None for value in values):
            continue
        if previous_altitude <= 200 and current_altitude <= 200 and previous_speed == 0 and current_speed > 0:
            return {
                "zero_timestamp": getattr(previous, "timestamp", ""),
                "movement_timestamp": getattr(current, "timestamp", ""),
                "movement_speed": current_speed,
                "movement_altitude": current_altitude,
            }
    return None


def event_times(client: Client, limiter: RateLimiter, flight_id: str) -> tuple[str, str]:
    response = limiter.call(
        lambda: client.historic.flight_events.get_full(
            flight_ids=[flight_id], event_types=["gate_departure", "takeoff"]
        ),
        f"flight events for {flight_id}",
    )
    data = list(response.data or [])
    events = list(getattr(data[0], "events", []) or []) if data else []
    gate_departure = ""
    takeoff = ""
    for event in events:
        event_type = getattr(event, "type", "") or ""
        timestamp = getattr(event, "timestamp", "") or ""
        if event_type == "gate_departure" and not gate_departure:
            gate_departure = timestamp
        elif event_type == "takeoff" and not takeoff:
            takeoff = timestamp
    return utc_string(gate_departure), utc_string(takeoff)


def collect_day(
    client: Client,
    limiter: RateLimiter,
    day_local: date,
    target_per_day: int,
    summary_limit: int,
) -> list[dict[str, object]]:
    start_utc, end_utc = local_day_bounds(day_local)
    response = limiter.call(
        lambda: client.flight_summary.get_full(
            airports=[AIRPORT_FILTER],
            flight_datetime_from=start_utc.strftime("%Y-%m-%dT%H:%M:%S"),
            flight_datetime_to=end_utc.strftime("%Y-%m-%dT%H:%M:%S"),
            sort="asc",
            limit=summary_limit,
        ),
        f"flight summary for {day_local.isoformat()}",
    )
    summaries = sorted(
        list(response.data or []),
        key=lambda flight: parse_timestamp(getattr(flight, "datetime_takeoff", ""))
        or datetime(2100, 1, 1, tzinfo=timezone.utc),
    )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for flight in summaries:
        if len(rows) >= target_per_day:
            break
        flight_id = getattr(flight, "fr24_id", "") or ""
        if not flight_id or flight_id in seen:
            continue
        seen.add(flight_id)

        try:
            tracks_response = limiter.call(
                lambda fid=flight_id: client.flight_tracks.get(flight_id=fid),
                f"flight tracks for {flight_id}",
            )
            blocks = list(tracks_response.data or [])
            track_points = list(getattr(blocks[0], "tracks", []) or []) if blocks else []
            movement = first_ground_movement(track_points)
            if not movement:
                continue
            gate_departure, takeoff_event = event_times(client, limiter, flight_id)
            summary_takeoff = utc_string(getattr(flight, "datetime_takeoff", "") or "")
            if not takeoff_event and not summary_takeoff:
                continue

            rows.append(
                {
                    "requested_day_local": day_local.isoformat(),
                    "flight_number": getattr(flight, "flight", "") or "",
                    "callsign": getattr(flight, "callsign", "") or "",
                    "aircraft_type": getattr(flight, "type", "") or "",
                    "registration": getattr(flight, "reg", "") or "",
                    "origin_iata": getattr(flight, "orig_iata", "") or "",
                    "origin_icao": getattr(flight, "orig_icao", "") or "",
                    "destination_iata": getattr(flight, "dest_iata", "") or "",
                    "destination_icao": getattr(flight, "dest_icao", "") or "",
                    "runway_takeoff": getattr(flight, "runway_takeoff", "") or "",
                    "first_groundspeed_zero_ts_utc": utc_string(movement["zero_timestamp"]),
                    "first_groundspeed_positive_ts_utc": utc_string(movement["movement_timestamp"]),
                    "first_groundspeed_positive_kt": movement["movement_speed"],
                    "first_groundspeed_positive_alt_ft": movement["movement_altitude"],
                    "gate_departure_event_utc": gate_departure,
                    "takeoff_event_utc": takeoff_event,
                    "takeoff_time_utc": summary_takeoff,
                    "fr24_id": flight_id,
                }
            )
            print(f"{day_local}: collected {len(rows)}/{target_per_day}")
        except Exception as exc:
            print(f"Skipping flight {flight_id}: {exc}")
    return rows


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    args = parse_args()
    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")
    token = os.environ.get("FR24_API_TOKEN")
    if not token:
        raise RuntimeError("Set FR24_API_TOKEN in the environment before collection.")

    limiter = RateLimiter(args.minimum_call_interval)
    day = args.start_date
    with Client(api_token=token) as client:
        while day <= args.end_date:
            rows = collect_day(client, limiter, day, args.target_per_day, args.summary_limit)
            append_rows(args.output, rows)
            day += timedelta(days=1)
    print(f"Raw departures saved to {args.output}")


if __name__ == "__main__":
    main()
