from __future__ import annotations
import sqlite3
import json
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import unicodedata
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")

DATE_FORMAT = "%Y-%m-%d"


# -------------------------
# Database helpers
# -------------------------
def connect_db(path: str = "numero_data.sqlite") -> sqlite3.Connection:
    """Open a sqlite3 connection to the given file path."""
    return sqlite3.connect(path)


def list_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame listing tables in the database."""
    return pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)


def table_info(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Return PRAGMA table_info for a table."""
    return pd.read_sql(f"PRAGMA table_info({table});", conn)


def sample_raw_json(conn: sqlite3.Connection, numero_film_id: int) -> Optional[dict]:
    """Return parsed JSON for a single film id (or None)."""
    row = pd.read_sql(
        "SELECT raw_json FROM sales_raw_data WHERE numero_film_id = ? LIMIT 1;",
        conn,
        params=(numero_film_id,),
    )
    if row.empty:
        return None
    raw_json_string = row.iloc[0, 0]
    try:
        return json.loads(raw_json_string)
    except json.JSONDecodeError:
        return None


# -------------------------
# Raw sales loader
# -------------------------
def load_raw_sales(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load raw sales table (numero_film_id, raw_json)."""
    df_raw = pd.read_sql("SELECT numero_film_id, raw_json FROM sales_raw_data;", conn)
    print(f"Loaded {len(df_raw)} raw film records.")
    return df_raw


# -------------------------
# JSON flattening helpers
# -------------------------
def _parse_week_start_date(week_start_date: str) -> Optional[datetime.date]:
    """Parse the week_start_date string into a date object or return None."""
    try:
        return datetime.datetime.strptime(week_start_date, DATE_FORMAT).date()
    except Exception:
        return None


def _process_single_film_record(film_id: int | str, raw_json_str: str) -> List[Dict]:
    """Flatten a single film's JSON structure into row-level dicts."""
    processed_rows: List[Dict] = []

    try:
        json_data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        # skip invalid JSON
        return processed_rows

    if not isinstance(json_data, dict):
        return processed_rows

    for week_start_date, week_content in json_data.items():
        if not isinstance(week_content, dict):
            continue

        week_dt = _parse_week_start_date(week_start_date)
        if week_dt is None:
            continue

        for cinema_row in week_content.get("rows", []):
            if not isinstance(cinema_row, dict):
                continue

            # Basic info
            state = cinema_row.get("state")
            state_id = cinema_row.get("stateId")
            region = cinema_row.get("region")
            region_id = cinema_row.get("regionId")
            city = cinema_row.get("city")
            city_id = cinema_row.get("cityId")
            theatre_name = cinema_row.get("theatre")
            theatre_id = cinema_row.get("theatreId")
            circuit_name = cinema_row.get("circuit")
            circuit_id = cinema_row.get("circuitId")
            cinema_rank_for_week = cinema_row.get("rank")

            # Release level info
            release_data = cinema_row.get("release", {}) or {}
            cumulative_gross = release_data.get("cumulativeBoxOffice")
            cumulative_admissions = release_data.get("cumulativePaidAdmissions")
            film_rank_at_cinema_for_week = release_data.get("thisWeekRank")
            films_in_cinema_this_week = release_data.get("thisWeekFilmCount")

            # Box office info
            box_office = cinema_row.get("boxOffice", {}) or {}
            week_summary = box_office.get("week", {}) or {}
            weekend_summary = box_office.get("weekend", {}) or {}

            week_gross_at_cinema = week_summary.get("gross")
            weekend_gross_at_cinema = weekend_summary.get("gross")

            for day_key, sales_data in box_office.items():
                if not (isinstance(day_key, str) and day_key.startswith("day")):
                    continue
                if not isinstance(sales_data, dict):
                    continue

                try:
                    day_num = int(day_key.replace("day", "")) - 1
                    if day_num < 0:
                        continue
                except Exception:
                    continue

                actual_sales_date = week_dt + datetime.timedelta(days=day_num)

                processed_rows.append(
                    {
                        "numero_film_id": film_id,
                        "week_start_date": week_start_date,
                        "actual_sales_date": actual_sales_date,
                        "gross_today": sales_data.get("today"),
                        "gross_yesterday": sales_data.get("yesterday"),
                        "paid_admissions": sales_data.get("paidAdmissions"),
                        "state": state,
                        "region": region,
                        "city": city,
                        "theatre_name": theatre_name,
                        "circuit_name": circuit_name,
                        "cinema_rank_for_week": cinema_rank_for_week,
                        "film_rank_at_cinema_for_week": film_rank_at_cinema_for_week,
                        "films_in_cinema_this_week": films_in_cinema_this_week,
                        "week_gross_at_cinema": week_gross_at_cinema,
                        "weekend_gross_at_cinema": weekend_gross_at_cinema,
                        "cumulative_gross_at_cinema": cumulative_gross,
                        "cumulative_admissions_at_cinema": cumulative_admissions,
                        "state_id": state_id,
                        "region_id": region_id,
                        "city_id": city_id,
                        "theatre_id": theatre_id,
                        "circuit_id": circuit_id,
                    }
                )

    return processed_rows


def flatten_sales_json(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten all raw JSON rows into a single tabular DataFrame."""
    processed_data: List[Dict] = []

    for raw_row in tqdm(df_raw.itertuples(index=False), total=len(df_raw), desc="Flattening sales JSON"):
        film_id = raw_row.numero_film_id
        raw_json_str = raw_row.raw_json
        try:
            film_rows = _process_single_film_record(film_id, raw_json_str)
            processed_data.extend(film_rows)
        except Exception:
            # continue on error for robustness
            continue

    print(f"Successfully processed {len(processed_data)} rows.")
    return pd.DataFrame(processed_data)


# -------------------------
# Post processing and joins
# -------------------------

def clean_title(title):
    """
    Produce a compact lowercase alphanumeric-only key for a title.
    Returns None for missing input.
    """
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return None
    s = str(title)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    return re.sub(r"[^a-z0-9]", "", s)

def make_title_label(title, max_len=60):
    """
    Produce a readable label for hover/visualization:
    - trims whitespace, collapses internal spaces, truncates with ellipsis.
    Returns None for missing input.
    """
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return None
    s = str(title).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s

def load_metadata_and_titles(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load film_metadata and indian_titles tables from database and add cleaned title columns."""
    df_titles = pd.read_sql("SELECT * FROM indian_titles;", conn)
    df_meta = pd.read_sql("SELECT * FROM film_metadata;", conn)

    # Ensure title column exists
    if 'title' not in df_titles.columns:
        df_titles['title'] = None
    if 'title' not in df_meta.columns:
        df_meta['title'] = None

    # Add cleaned and label columns (preserve original title)
    df_titles['title_clean'] = df_titles['title'].apply(clean_title)
    df_titles['title_label'] = df_titles['title'].apply(make_title_label)

    df_meta['title_clean'] = df_meta['title'].apply(clean_title)
    df_meta['title_label'] = df_meta['title'].apply(make_title_label)

    return df_meta, df_titles

def join_sales_metadata(sales_df: pd.DataFrame, df_meta: pd.DataFrame, df_titles: pd.DataFrame) -> pd.DataFrame:
    """Join sales to metadata and titles, and add derived date columns."""
    for df in (df_meta, df_titles):
        if 'title' not in df.columns:
            df['title'] = None
        if 'title_clean' not in df.columns:
            df['title_clean'] = df['title'].apply(clean_title)
        if 'title_label' not in df.columns:
            df['title_label'] = df['title'].apply(make_title_label)

    # merge metadata and titles on title_clean
    meta_plus_titles = df_meta.merge(
        df_titles[['title_clean', 'title', 'title_label']].rename(columns={'title': 'title_from_titles', 'title': 'title_from_titles'}),
        on='title_clean',
        how='left',
        suffixes=('', '_titles')
    )

    # merge metadata
    df = sales_df.merge(meta_plus_titles, on='numero_film_id', how='left')

    # derived date columns and cleanups 
    df["actual_sales_date"] = pd.to_datetime(df["actual_sales_date"])
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["month"] = df["actual_sales_date"].dt.to_period("M").dt.to_timestamp()
    df["dow"] = df["actual_sales_date"].dt.dayofweek
    df["week_offset"] = (df["actual_sales_date"] - df["week_start_date"]).dt.days
    df["is_weekend_numero"] = df["week_offset"].between(0, 3)
    df["theatre_name"] = df["theatre_name"].str.replace(r"\s\d+$", "", regex=True).str.strip()

    return df


def join_sales_metadata(sales_df: pd.DataFrame, df_meta: pd.DataFrame, df_titles: pd.DataFrame) -> pd.DataFrame:
    """Join sales to metadata and titles, and add derived date columns."""
    df_sales_meta = sales_df.merge(df_meta, on="numero_film_id", how="left")
    df = df_sales_meta.merge(df_titles, on="title", how="left", suffixes=("", "_titles"))

    df["actual_sales_date"] = pd.to_datetime(df["actual_sales_date"])
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["month"] = df["actual_sales_date"].dt.to_period("M").dt.to_timestamp()
    df["dow"] = df["actual_sales_date"].dt.dayofweek
    df["week_offset"] = (df["actual_sales_date"] - df["week_start_date"]).dt.days
    df["is_weekend_numero"] = df["week_offset"].between(0, 3)
    df["theatre_name"] = df["theatre_name"].str.replace(r"\s\d+$", "", regex=True).str.strip()
    return df

def extract_longest_continuous_run(df: pd.DataFrame, gap_days: int = 7) -> pd.DataFrame:
    """Identify continuous segments per film and return df_run containing the longest continuous run."""
    # ensure date column is datetime
    df['actual_sales_date'] = pd.to_datetime(df['actual_sales_date'])
    dataset_end_date = df['actual_sales_date'].max()

    # film-level first/last and duration/event flags (keeps your film_dates variable)
    film_dates = (
        df.groupby('numero_film_id', as_index=False)
          .agg(first_date=('actual_sales_date', 'min'), last_date=('actual_sales_date', 'max'))
    )
    film_dates['duration_days'] = (film_dates['last_date'] - film_dates['first_date']).dt.days + 1
    film_dates['event_observed'] = (film_dates['last_date'] < dataset_end_date).astype(int)

    # prepare per-day rows per film
    tmp = (
        df[['numero_film_id', 'actual_sales_date']]
        .drop_duplicates()
        .sort_values(['numero_film_id', 'actual_sales_date'])
        .copy()
    )

    tmp['prev_date'] = tmp.groupby('numero_film_id')['actual_sales_date'].shift(1)
    tmp['gap'] = (tmp['actual_sales_date'] - tmp['prev_date']).dt.days.fillna(0)
    tmp['new_segment'] = (tmp['gap'] > gap_days).astype(int)
    tmp['segment_id'] = tmp.groupby('numero_film_id')['new_segment'].cumsum()

    segments = (
        tmp.groupby(['numero_film_id', 'segment_id'], as_index=False)
           .agg(seg_start=('actual_sales_date', 'min'),
                seg_end=('actual_sales_date', 'max'))
    )

    segments['seg_days'] = (segments['seg_end'] - segments['seg_start']).dt.days + 1

    longest = (
        segments.groupby('numero_film_id')['seg_days']
                .max()
                .rename('run_days_longest')
                .reset_index()
    )

    # remove any existing run_days_longest columns in df to avoid duplicates
    cols_to_drop = [c for c in df.columns if c.startswith('run_days_longest')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.merge(longest, on='numero_film_id', how='left')

    # attach segment_id for each actual_sales_date
    df = df.merge(tmp[['numero_film_id', 'actual_sales_date', 'segment_id']],
                  on=['numero_film_id', 'actual_sales_date'], how='left')

    # find the longest segment id per film
    longest_seg = (
        segments.sort_values(['numero_film_id', 'seg_days'], ascending=[True, False])
                .drop_duplicates('numero_film_id')[['numero_film_id', 'segment_id']]
                .rename(columns={'segment_id': 'longest_segment'})
    )

    df = df.merge(longest_seg, on='numero_film_id', how='left')

    # filtered dataset: only rows that belong to the longest continuous run
    df_run = df[df['segment_id'] == df['longest_segment']].copy()

    # ensure datetime and compute run-relative fields
    df_run['actual_sales_date'] = pd.to_datetime(df_run['actual_sales_date'])
    df_run['first_date'] = df_run.groupby('numero_film_id')['actual_sales_date'].transform('min')
    df_run['day'] = (df_run['actual_sales_date'] - df_run['first_date']).dt.days + 1
    df_run['week'] = (df_run['day'] - 1) // 7 + 1

    return df_run, longest, segments, tmp, film_dates

# -------------------------
# Convenience functions
# -------------------------
def save_df_to_csv(df: pd.DataFrame, path: str = "sales_processed.csv") -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)


def load_processed_csv(path: str = "sales_processed.csv") -> pd.DataFrame:
    """Load processed CSV if present."""
    return pd.read_csv(path)


# -------------------------
# Script entrypoint
# -------------------------
def main(db_path: str = "numero_data.sqlite", output_csv: str = "sales_processed.csv") -> None:
    conn = connect_db(db_path)
    df_raw = load_raw_sales(conn)
    if df_raw.empty:
        print("No raw sales data found.")
        return

    df_clean = flatten_sales_json(df_raw)
    save_df_to_csv(df_clean, output_csv)
    print(df_clean.head())
    print(f"Saved processed data to {output_csv}")


if __name__ == "__main__":
    main()
