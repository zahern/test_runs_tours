from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sqlite3
import time
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from urllib3.exceptions import NotOpenSSLWarning


# ============================================================
# CONFIG
# ============================================================

API_RECORDS_URL = (
    "https://data.brisbane.qld.gov.au/api/explore/v2.1/catalog/datasets/"
    "traffic-data-at-intersection/records"
)

# --- LOGIN (kept, as you requested) ---
# Note: Old portal domain; may 403 now. Safe to ignore for OpenDataSoft API.
USE_LOGIN = False
LOGIN_URL = "https://www.data.brisbane.qld.gov.au/data/data/login_generic?came_from=/data/user/logged_in"
LOGIN_DATA = {"login": "", "password": "", "remember": ""}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
Z_location = Path(r"Z:/")
OUT_ROOT = Z_location / "BCC Volume Data"

POLL_SECONDS = 20

# Portal constraint: limit must be <= 100 per request
PAGE_LIMIT = 100

# How many newest rows to fetch per poll (paging via offset). Increase if needed.
TOTAL_ROWS_TO_FETCH = 9999


# ============================================================
# COLUMN DISCOVERY
# ============================================================

DS_MF_RF_RE = re.compile(r"^(ds|mf|rf)(\d+)$")
BASE_COLS = ["dbid", "recorded_date", "recorded_time", "ct", "link_plan", "married", "ss", "tsc", "lane"]


def split_recorded(recorded: Any) -> Tuple[str, str]:
    if recorded is None:
        return "", ""
    s = str(recorded)
    if "T" in s:
        d, t = s.split("T", 1)
        return d.strip(), t.strip()
    return s.strip(), ""


def discover_dyn_cols(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Discover any ds*/mf*/rf* columns present in returned JSON.
    Returns ordered list: ds1,mf1,rf1, ds2,mf2,rf2, ...
    """
    idx_set = set()
    kind_set = set()

    for r in rows:
        for k in r.keys():
            m = DS_MF_RF_RE.match(k)
            if m:
                kind_set.add(m.group(1))
                idx_set.add(int(m.group(2)))

    indices = sorted(idx_set)
    dyn: List[str] = []
    for i in indices:
        for kind in ("ds", "mf", "rf"):
            if kind in kind_set:
                dyn.append(f"{kind}{i}")
    return dyn


# ============================================================
# DIRECTORY LAYOUT
# ============================================================

def month_day_dirs(now: dt.datetime) -> Path:
    month_label = now.strftime("%B, %Y")   # e.g., March, 2026
    day_label = now.strftime("%d-%m-%Y")   # e.g., 05-03-2026
    day_dir = OUT_ROOT / f"BCC Vol {month_label}" / f"BCC Vol {day_label}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def columns_meta_path(day_dir: Path) -> Path:
    return day_dir / "columns.json"


def seen_db_path(day_dir: Path) -> Path:
    return day_dir / "seen_index.sqlite"


# ============================================================
# SEEN INDEX (DE-DUP) PER DAY
# ============================================================

def init_seen_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                dbid TEXT PRIMARY KEY
            )
            """
        )
        con.commit()
    finally:
        con.close()


def is_new_dbid(db_path: Path, dbid: Any) -> bool:
    if dbid is None:
        return False
    key = str(dbid)

    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("INSERT OR IGNORE INTO seen(dbid) VALUES (?)", (key,))
        con.commit()
        return cur.rowcount == 1
    finally:
        con.close()


# ============================================================
# CSV UTILITIES
# ============================================================

def read_existing_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite_csv_with_new_header(csv_path: Path, new_header: List[str]) -> None:
    if not csv_path.exists():
        return
    rows = read_existing_csv_rows(csv_path)
    for r in rows:
        for col in new_header:
            r.setdefault(col, "")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_header)
        w.writeheader()
        w.writerows(rows)


def append_rows(csv_path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    is_new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new_file:
            w.writeheader()
        w.writerows(rows)


# ============================================================
# HEADER MANAGEMENT (PER DAY)
# ============================================================

def load_or_init_header(day_dir: Path, dyn_cols_from_batch: List[str]) -> List[str]:
    meta_path = columns_meta_path(day_dir)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        header = meta.get("header", [])
        if not isinstance(header, list) or not header:
            header = BASE_COLS + dyn_cols_from_batch
    else:
        header = BASE_COLS + dyn_cols_from_batch

    # Expand header if new dyn cols appear
    header_set = set(header)
    changed = False
    for c in dyn_cols_from_batch:
        if c not in header_set:
            header.append(c)
            header_set.add(c)
            changed = True

    if (not meta_path.exists()) or changed:
        meta_path.write_text(json.dumps({"header": header}, indent=2), encoding="utf-8")

        # If header expanded, rewrite ALL per-tsc CSVs in the day folder once
        if changed:
            for csv_path in day_dir.glob("TSC_*.csv"):
                rewrite_csv_with_new_header(csv_path, header)

    return header


# ============================================================
# API FETCH
# ============================================================

def do_login(session: requests.Session) -> None:
    try:
        r = session.post(LOGIN_URL, data=LOGIN_DATA, headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            print(f"Login warning: status {r.status_code} at login URL (safe to ignore for API)")
    except Exception as e:
        print("Login warning:", repr(e))


def fetch_rows_paged(session: requests.Session, total: int) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0

    while len(all_rows) < total:
        this_limit = min(PAGE_LIMIT, total - len(all_rows))
        params = {
            "order_by": "recorded desc",
            "limit": this_limit,
            "offset": offset,
        }
        r = session.get(API_RECORDS_URL, headers=HEADERS, params=params, timeout=30)

        if r.status_code >= 400:
            try:
                print("API error body:", r.json())
            except Exception:
                print("API error body (text):", r.text[:1000])
            r.raise_for_status()

        payload = r.json()
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            break

        batch = [x for x in results if isinstance(x, dict)]
        all_rows.extend(batch)
        offset += len(batch)

        if len(batch) < this_limit:
            break

    return all_rows


# ============================================================
# NORMALIZE ROW FOR CSV
# ============================================================

def normalize_row(r: Dict[str, Any], header: List[str]) -> Dict[str, Any]:
    rec_date, rec_time = split_recorded(r.get("recorded"))
    out: Dict[str, Any] = {col: "" for col in header}

    out["dbid"] = r.get("dbid", "")
    out["recorded_date"] = rec_date
    out["recorded_time"] = rec_time
    out["ct"] = r.get("ct", "")
    out["link_plan"] = r.get("link_plan", "")
    out["married"] = r.get("married", "")
    out["ss"] = r.get("ss", "")
    out["tsc"] = r.get("tsc", "")
    out["lane"] = r.get("lane", "")

    for col in header:
        if col.startswith(("ds", "mf", "rf")):
            out[col] = r.get(col, "")

    return out


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

    current_day_key: str | None = None
    day_dir: Path | None = None
    seen_db: Path | None = None
    header: List[str] = []

    with requests.Session() as s:
        if USE_LOGIN:
            do_login(s)

        while True:
            try:
                now = dt.datetime.now()
                new_day_dir = month_day_dirs(now)
                today_key = new_day_dir.name  # "BCC Vol dd-mm-YYYY"

                # Day rollover
                if current_day_key != today_key:
                    current_day_key = today_key
                    day_dir = new_day_dir
                    seen_db = seen_db_path(day_dir)
                    init_seen_db(seen_db)
                    header = []
                    print(f"\n=== Writing to day folder: {day_dir} ===")

                assert day_dir is not None and seen_db is not None

                rows = fetch_rows_paged(s, TOTAL_ROWS_TO_FETCH)

                # Build/expand header based on what appears in the batch
                dyn_cols = discover_dyn_cols(rows)
                header = load_or_init_header(day_dir, dyn_cols)

                # Group NEW rows per tsc
                per_tsc: Dict[str, List[Dict[str, Any]]] = {}
                new_rows_total = 0

                for r in rows:
                    dbid = r.get("dbid")
                    if not is_new_dbid(seen_db, dbid):
                        continue  # already written today

                    tsc = str(r.get("tsc", "")).strip()
                    if not tsc:
                        tsc = "UNKNOWN"

                    per_tsc.setdefault(tsc, []).append(normalize_row(r, header))
                    new_rows_total += 1

                # Append to each per-tsc CSV
                for tsc, out_rows in per_tsc.items():
                    csv_path = day_dir / f"TSC_{tsc}.csv"
                    append_rows(csv_path, header, out_rows)

                if new_rows_total:
                    print(f"Appended {new_rows_total} new rows across {len(per_tsc)} TSC files.")
                else:
                    print("No new rows (today).")

            except Exception as e:
                print("Error:", repr(e))

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
