"""
Stage models for the PYRAM pipeline.

This module supplies the three pieces of the pipeline that sit between the
synthetic population and destination/mode choice:

- ``CDAPModel``: household coordinated daily activity patterns (H / N / M)
  plus joint tours, estimated from HHTS.
- ``PurposeTimingModel``: empirical joint distribution of departure time and
  duration per purpose, estimated from HHTS and applied to synthetic activity
  episodes.
- ``build_trip_chains``: assembles activity episodes into a trip list with
  tour/trip identifiers.

Estimation is empirical (survey resampling) rather than a parametric maximum
likelihood fit. The HHTS extract used here does not support a stable
multi-state CDAP fit, and resampling keeps the observed joint distributions
(start time x duration, non-mandatory purpose mix, joint-tour incidence) intact
instead of imposing an assumed functional form.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

CDAP_LABELS = {0: "H", 1: "N", 2: "M"}
CDAP_INDEX = {v: k for k, v in CDAP_LABELS.items()}

MANDATORY_PURPOSES = ("work", "edu")
NON_MANDATORY_PURPOSES = ("shopping", "social", "escort", "other")

_START_COLUMNS = ("STARTIME", "start_time", "START_TIME", "depart_time", "DEPART_TIME")
_END_COLUMNS = ("ENDTIME", "end_time", "END_TIME", "arrive_time", "ARRIVE_TIME")
_DURATION_COLUMNS = ("duration_min", "DURATION", "duration")
_PERSON_COLUMNS = ("person_id", "PERSID", "PERSID_")
_PURPOSE_COLUMNS = ("purpose", "PURPOSE", "trip_purpose", "TRIP_PURPOSE",
                    "tour_purpose", "TOUR_PURPOSE")


def _first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column present in ``df``."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalise_purpose(value: Any) -> str:
    """Map an HHTS/pyram purpose label onto the pipeline's purpose groups."""
    if value is None:
        return "other"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {1: "work", 2: "edu", 3: "shopping", 4: "social", 5: "escort"}.get(
            int(value), "other"
        )

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return "other"
    if text in {"home", "h", "return", "return_home"}:
        return "home"
    if text.startswith(("work", "biz", "business", "commut")):
        return "work"
    if text.startswith(("edu", "school", "univer", "study", "studying", "childcare", "tafe")):
        return "edu"
    if text.startswith(("shop", "buy", "grocer")):
        return "shopping"
    if text.startswith(("soc", "visit", "leis", "rec", "entertain", "sport")):
        return "social"
    if text.startswith(("escort", "esc", "pick", "drop")):
        return "escort"
    return "other"


def _pattern_from_purposes(purposes: pd.Series) -> str:
    """Classify a person-day as Home / Non-mandatory / Mandatory."""
    groups = {normalise_purpose(p) for p in purposes}
    groups.discard("home")
    if groups & set(MANDATORY_PURPOSES):
        return "M"
    if groups:
        return "N"
    return "H"


def _role_of(row: pd.Series) -> str:
    """Derive a person role key from HHTS person attributes."""
    ptype = row.get("ptype") if hasattr(row, "get") else None
    if isinstance(ptype, str) and ptype.strip() and ptype.strip().lower() != "nan":
        return ptype.strip().lower().replace(" ", "_")

    workstat = pd.to_numeric(pd.Series([row.get("WORKSTAT", 0)]), errors="coerce").iloc[0]
    studying = pd.to_numeric(pd.Series([row.get("STUDYING", 0)]), errors="coerce").iloc[0]
    worker = bool(pd.notna(workstat) and workstat > 0)
    student = bool(pd.notna(studying) and studying > 0)

    if worker and student:
        return "worker_student"
    if worker:
        return "worker"
    if student:
        return "student"
    return "other"


def _role_array(persons: pd.DataFrame) -> np.ndarray:
    """Vectorised :func:`_role_of` over a whole person table.

    ``expand_daily_patterns`` needs one role per person; deriving it row by row
    materialises a Series per person, which costs gigabytes at synthetic
    population scale.
    """
    n = len(persons)
    roles = np.full(n, "other", dtype=object)

    worker = np.zeros(n, dtype=bool)
    student = np.zeros(n, dtype=bool)
    if "WORKSTAT" in persons.columns:
        worker = (
            pd.to_numeric(persons["WORKSTAT"], errors="coerce").fillna(0).to_numpy() > 0
        )
    if "STUDYING" in persons.columns:
        student = (
            pd.to_numeric(persons["STUDYING"], errors="coerce").fillna(0).to_numpy() > 0
        )

    roles[worker & student] = "worker_student"
    roles[worker & ~student] = "worker"
    roles[~worker & student] = "student"

    if "ptype" in persons.columns:
        ptype = (
            persons["ptype"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        has_ptype = ptype.ne("") & ptype.ne("nan") & ptype.ne("none")
        if has_ptype.any():
            roles[has_ptype.to_numpy()] = ptype[has_ptype].to_numpy()
    return roles


class CDAPModel:
    """
    Household coordinated daily activity pattern model.

    States follow the standard CDAP convention:

    - ``H``: home all day (no out-of-home activity)
    - ``N``: non-mandatory activities only
    - ``M``: at least one mandatory activity (work or education)

    The individual pattern probabilities are estimated by person segment from
    HHTS, then a household coordination step elects joint non-mandatory tours
    for households that the survey shows making them.
    """

    PATTERNS = ("H", "N", "M")

    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.pattern_probabilities = pd.DataFrame(columns=list(self.PATTERNS))
        self.global_probabilities = pd.Series(
            [1.0 / 3.0] * 3, index=list(self.PATTERNS), dtype=float
        )
        self.nonmandatory_purpose_pmf = pd.Series(
            [1.0 / len(NON_MANDATORY_PURPOSES)] * len(NON_MANDATORY_PURPOSES),
            index=list(NON_MANDATORY_PURPOSES),
            dtype=float,
        )
        self.nonmandatory_count_pmf: Dict[int, float] = {1: 1.0}
        self.joint_tour_rate = 0.0
        self.joint_tour_purpose_pmf = pd.Series(dtype=float)
        self.max_activities = int(self.config.get("max_acts_per_person", 6))
        self.fitted = False

    # ── Estimation ──────────────────────────────────────────────────────────

    def estimate(
        self,
        persons: pd.DataFrame,
        trips: pd.DataFrame,
        hh_col: str = "hhid",
    ) -> "CDAPModel":
        """Estimate patterns, purpose mix and joint-tour incidence from HHTS."""
        if persons is None or len(persons) == 0:
            raise ValueError("CDAPModel.estimate requires a non-empty persons frame")
        if trips is None or len(trips) == 0:
            raise ValueError("CDAPModel.estimate requires a non-empty trips frame")

        person_col = _first_column(trips, _PERSON_COLUMNS)
        purpose_col = _first_column(trips, _PURPOSE_COLUMNS)
        if person_col is None or purpose_col is None:
            raise ValueError(
                "HHTS trips must provide a person id column and a purpose column; "
                f"found columns {list(trips.columns)[:12]}"
            )

        obs = trips[[person_col, purpose_col]].rename(
            columns={person_col: "person_id", purpose_col: "purpose"}
        )
        obs["person_id"] = obs["person_id"].astype(str)
        obs["purpose_group"] = obs["purpose"].map(normalise_purpose)

        observed = obs.groupby("person_id")["purpose_group"].apply(_pattern_from_purposes)

        people = persons.copy()
        people["person_id"] = people["person_id"].astype(str)
        people["_pattern"] = people["person_id"].map(observed).fillna("H")
        people["_segment"] = people.apply(_role_of, axis=1)

        counts = pd.crosstab(people["_segment"], people["_pattern"])
        counts = counts.reindex(columns=list(self.PATTERNS), fill_value=0)
        # Add-one smoothing keeps unobserved states reachable for rare segments.
        self.pattern_probabilities = (counts + 1.0).div(counts.sum(axis=1) + 3.0, axis=0)

        global_counts = people["_pattern"].value_counts().reindex(
            list(self.PATTERNS), fill_value=0
        ).astype(float) + 1.0
        self.global_probabilities = global_counts / global_counts.sum()

        # Non-mandatory purpose mix and per-person activity counts.
        nonmandatory = obs[obs["purpose_group"].isin(NON_MANDATORY_PURPOSES)]
        if len(nonmandatory):
            mix = nonmandatory["purpose_group"].value_counts(normalize=True)
            mix = mix.reindex(list(NON_MANDATORY_PURPOSES), fill_value=0.0) + 1e-6
            self.nonmandatory_purpose_pmf = mix / mix.sum()

            per_person = (
                nonmandatory.groupby("person_id").size().clip(upper=self.max_activities)
            )
            count_dist = per_person.value_counts(normalize=True)
            self.nonmandatory_count_pmf = {
                int(k): float(v) for k, v in count_dist.sort_index().items() if int(k) > 0
            } or {1: 1.0}

        # Joint tours: households where two or more members share a
        # non-mandatory purpose on the survey day.
        if hh_col in people.columns:
            hh_map = people.dropna(subset=[hh_col]).drop_duplicates("person_id").set_index(
                "person_id"
            )[hh_col]
            merged = obs.join(hh_map.rename("_hhid"), on="person_id", how="inner")
            merged = merged[merged["purpose_group"].isin(NON_MANDATORY_PURPOSES)]

            if len(merged):
                shared = merged.groupby(["_hhid", "purpose_group"])["person_id"].nunique()
                shared = shared[shared >= 2]
                n_households = people[hh_col].nunique()

                if len(shared) and n_households:
                    households_with_joint = shared.reset_index()["_hhid"].nunique()
                    self.joint_tour_rate = float(
                        min(1.0, households_with_joint / max(n_households, 1))
                    )
                    self.joint_tour_purpose_pmf = (
                        shared.reset_index()["purpose_group"].value_counts(normalize=True)
                    )

        self.fitted = True
        return self

    # ── Simulation ──────────────────────────────────────────────────────────

    def _probabilities_for(self, segment: str) -> np.ndarray:
        if segment in self.pattern_probabilities.index:
            probs = self.pattern_probabilities.loc[segment].to_numpy(dtype=float)
        else:
            probs = self.global_probabilities.to_numpy(dtype=float)
        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum()

    def simulate(self, persons: pd.DataFrame) -> pd.DataFrame:
        """Draw a daily activity pattern for each synthetic person."""
        if not self.fitted:
            raise RuntimeError("CDAPModel must be estimated before simulate()")

        people = persons.copy()
        if "person_id" not in people.columns:
            raise ValueError("synthetic population requires a person_id column")

        segments = people.apply(_role_of, axis=1).astype(str).to_numpy()
        probabilities = np.vstack([self._probabilities_for(s) for s in segments])
        draws = np.array(
            [self.rng.choice(3, p=p) for p in probabilities], dtype=int
        )

        patterns = pd.DataFrame(
            {
                "person_id": people["person_id"].astype(str).to_numpy(),
                "segment": segments,
                "cdap": [CDAP_LABELS[int(d)] for d in draws],
                "p_H": probabilities[:, 0],
                "p_N": probabilities[:, 1],
                "p_M": probabilities[:, 2],
            }
        )
        if "hhid" in people.columns:
            patterns["hhid"] = people["hhid"].to_numpy()
        return patterns

    def coordinate_households(
        self,
        patterns: pd.DataFrame,
        persons: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Elect joint non-mandatory tours within households.

        Only households with at least two non-home members can host a joint
        tour; the incidence rate and purpose mix come from the HHTS estimate.
        """
        columns = ["tour_id", "hhid", "purpose", "participants", "n_participants"]
        if patterns is None or len(patterns) == 0 or "hhid" not in patterns.columns:
            return pd.DataFrame(columns=columns)
        if self.joint_tour_rate <= 0.0:
            return pd.DataFrame(columns=columns)

        purposes = (
            self.joint_tour_purpose_pmf
            if len(self.joint_tour_purpose_pmf)
            else self.nonmandatory_purpose_pmf
        )
        purpose_values = purposes.index.astype(str).to_numpy()
        purpose_probs = purposes.to_numpy(dtype=float)
        purpose_probs = purpose_probs / purpose_probs.sum()

        tours: List[Dict[str, Any]] = []
        active = patterns[patterns["cdap"] != "H"]
        for index, (hhid, members) in enumerate(active.groupby("hhid", sort=False)):
            if len(members) < 2:
                continue
            if self.rng.random() > self.joint_tour_rate:
                continue
            purpose = str(self.rng.choice(purpose_values, p=purpose_probs))
            participants = members["person_id"].astype(str).tolist()
            tours.append(
                {
                    "tour_id": f"J{index + 1}",
                    "hhid": hhid,
                    "purpose": purpose,
                    "participants": "|".join(participants),
                    "n_participants": len(participants),
                }
            )

        return pd.DataFrame(tours, columns=columns)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "seed": self.seed,
                    "pattern_probabilities": self.pattern_probabilities,
                    "global_probabilities": self.global_probabilities,
                    "nonmandatory_purpose_pmf": self.nonmandatory_purpose_pmf,
                    "nonmandatory_count_pmf": self.nonmandatory_count_pmf,
                    "joint_tour_rate": self.joint_tour_rate,
                    "joint_tour_purpose_pmf": self.joint_tour_purpose_pmf,
                    "max_activities": self.max_activities,
                    "fitted": self.fitted,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CDAPModel":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        model = cls(data.get("config", {}), data.get("seed", 42))
        model.pattern_probabilities = data["pattern_probabilities"]
        model.global_probabilities = data["global_probabilities"]
        model.nonmandatory_purpose_pmf = data["nonmandatory_purpose_pmf"]
        model.nonmandatory_count_pmf = data.get("nonmandatory_count_pmf", {1: 1.0})
        model.joint_tour_rate = data.get("joint_tour_rate", 0.0)
        model.joint_tour_purpose_pmf = data.get(
            "joint_tour_purpose_pmf", pd.Series(dtype=float)
        )
        model.max_activities = data.get("max_activities", 6)
        model.fitted = data.get("fitted", True)
        return model


def expand_daily_patterns(
    patterns: pd.DataFrame,
    persons: Optional[pd.DataFrame] = None,
    joint_tours: Optional[pd.DataFrame] = None,
    cdap_model: Optional[CDAPModel] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Expand daily patterns into activity episodes.

    Mandatory days yield a work or education episode (depending on the person
    role) plus any joint activity; non-mandatory days yield one or more
    non-mandatory episodes drawn from the estimated purpose mix.
    """
    rng = np.random.default_rng(seed)
    model = cdap_model or CDAPModel(seed=seed)

    purposes = model.nonmandatory_purpose_pmf
    purpose_values = purposes.index.astype(str).to_numpy()
    purpose_probs = purposes.to_numpy(dtype=float)
    purpose_probs = purpose_probs / purpose_probs.sum()

    counts = model.nonmandatory_count_pmf or {1: 1.0}
    count_values = np.array(sorted(counts), dtype=int)
    count_probs = np.array([counts[int(k)] for k in count_values], dtype=float)
    count_probs = count_probs / count_probs.sum()

    roles: Dict[str, str] = {}
    if persons is not None and "person_id" in persons.columns:
        roles = dict(
            zip(
                persons["person_id"].astype(str).to_numpy(),
                _role_array(persons),
            )
        )

    joint_purpose: Dict[str, str] = {}
    if joint_tours is not None and len(joint_tours):
        for tour in joint_tours.itertuples():
            for participant in str(tour.participants).split("|"):
                if participant:
                    joint_purpose.setdefault(participant, str(tour.purpose))

    rows: List[Dict[str, Any]] = []
    for pattern in patterns.itertuples():
        person_id = str(pattern.person_id)
        state = str(pattern.cdap)
        activities: List[str] = []

        if state == "M":
            role = roles.get(person_id, "other")
            activities.append("edu" if "student" in role else "work")
        elif state == "N":
            n_acts = int(rng.choice(count_values, p=count_probs))
            activities.extend(
                str(rng.choice(purpose_values, size=n_acts, p=purpose_probs))
            )

        joint = joint_purpose.get(person_id)
        if joint is not None and state != "H":
            activities.append(joint)

        activities = activities[: max(model.max_activities, 1)]
        for seq, activity in enumerate(activities):
            rows.append(
                {
                    "person_id": person_id,
                    "activity_type": activity,
                    "seq": seq,
                    "joint": int(joint is not None and activity == joint),
                    "cdap": state,
                }
            )

    return pd.DataFrame(
        rows, columns=["person_id", "activity_type", "seq", "joint", "cdap"]
    )


class PurposeTimingModel:
    """
    Empirical departure-time and duration model.

    For each purpose the model stores the observed (start hour, duration hour)
    pairs from HHTS and draws from them jointly, so the correlation between
    departure time and duration is preserved. Episodes are then sequenced per
    person so that they do not overlap.
    """

    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.purpose_samples: Dict[str, np.ndarray] = {}
        self.pooled_samples = np.empty((0, 2), dtype=float)
        self.fitted = False

    def estimate(self, trips: pd.DataFrame) -> "PurposeTimingModel":
        """Estimate empirical start-time/duration pairs from HHTS trips."""
        if trips is None or len(trips) == 0:
            raise ValueError("PurposeTimingModel.estimate requires a non-empty trips frame")

        start_col = _first_column(trips, _START_COLUMNS)
        end_col = _first_column(trips, _END_COLUMNS)
        duration_col = _first_column(trips, _DURATION_COLUMNS)
        purpose_col = _first_column(trips, _PURPOSE_COLUMNS)

        if start_col is None and duration_col is None:
            raise ValueError(
                "HHTS trips must provide a start time and a duration (or end time) column"
            )
        if purpose_col is None:
            raise ValueError("HHTS trips must provide a purpose column")

        data = pd.DataFrame(index=trips.index)
        data["purpose"] = trips[purpose_col].astype(str).map(normalise_purpose)

        if start_col is not None:
            start = pd.to_numeric(trips[start_col], errors="coerce")
            if start.dropna().gt(1440 * 2).any():  # values recorded in seconds
                start = start / 60.0
            data["start_hour"] = start / 60.0
        else:
            data["start_hour"] = np.nan

        if duration_col is not None:
            duration = pd.to_numeric(trips[duration_col], errors="coerce")
            if duration.dropna().gt(24 * 2).any():  # minutes
                duration = duration / 60.0
            data["duration_hour"] = duration
        elif end_col is not None:
            end = pd.to_numeric(trips[end_col], errors="coerce")
            if end.dropna().gt(1440 * 2).any():
                end = end / 60.0
            data["duration_hour"] = end / 60.0 - data["start_hour"]
        else:
            data["duration_hour"] = np.nan

        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) == 0:
            raise ValueError("no usable start-time/duration pairs found in HHTS trips")

        data["start_hour"] = data["start_hour"].clip(0.0, 24.0)
        data["duration_hour"] = data["duration_hour"].clip(0.05, 24.0)
        data = data[data["purpose"] != "home"]
        if len(data) == 0:
            raise ValueError("no out-of-home trips with usable times found")

        self.pooled_samples = data[["start_hour", "duration_hour"]].to_numpy(dtype=float)
        for purpose, group in data.groupby("purpose"):
            samples = group[["start_hour", "duration_hour"]].to_numpy(dtype=float)
            if len(samples) >= 5:
                self.purpose_samples[str(purpose)] = samples

        self.fitted = True
        return self

    def predict(self, activities: pd.DataFrame) -> pd.DataFrame:
        """Assign start times and durations to activity episodes."""
        if not self.fitted:
            raise RuntimeError("PurposeTimingModel must be estimated before predict()")

        scheduled = activities.copy()
        if len(scheduled) == 0:
            scheduled["start_time"] = pd.Series(dtype=float)
            scheduled["duration"] = pd.Series(dtype=float)
            scheduled["end_time"] = pd.Series(dtype=float)
            return scheduled

        start_times = np.empty(len(scheduled), dtype=float)
        durations = np.empty(len(scheduled), dtype=float)

        for position, purpose in enumerate(scheduled["activity_type"].astype(str)):
            samples = self.purpose_samples.get(purpose, self.pooled_samples)
            draw = samples[int(self.rng.integers(0, len(samples)))]
            jitter = self.rng.normal(0.0, 0.1)
            start_times[position] = np.clip(draw[0] + jitter, 0.0, 23.75)
            durations[position] = np.clip(
                draw[1] * float(np.exp(self.rng.normal(0.0, 0.1))), 0.25, 24.0
            )

        scheduled["start_time"] = start_times
        scheduled["duration"] = durations
        scheduled["end_time"] = np.minimum(scheduled["start_time"] + scheduled["duration"], 24.0)

        return _sequence_episodes(scheduled)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "seed": self.seed,
                    "purpose_samples": self.purpose_samples,
                    "pooled_samples": self.pooled_samples,
                    "fitted": self.fitted,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PurposeTimingModel":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        model = cls(data.get("config", {}), data.get("seed", 42))
        model.purpose_samples = data.get("purpose_samples", {})
        model.pooled_samples = data.get("pooled_samples", np.empty((0, 2)))
        model.fitted = data.get("fitted", True)
        return model


def _sequence_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Remove within-person overlap by pushing later activities back in time."""
    if "person_id" not in episodes.columns or len(episodes) == 0:
        return episodes

    ordered = episodes.sort_values(["person_id", "start_time"], kind="stable").copy()
    for _, indices in ordered.groupby("person_id", sort=False).groups.items():
        cursor = 0.0
        for index in indices:
            start = max(float(ordered.at[index, "start_time"]), cursor)
            duration = float(ordered.at[index, "duration"])
            if start > 23.5:
                start = 23.5
            duration = min(duration, 24.0 - start)
            if duration < 0.25:
                start = max(0.0, 24.0 - 0.25)
                duration = 24.0 - start
            ordered.at[index, "start_time"] = round(start, 4)
            ordered.at[index, "duration"] = round(duration, 4)
            ordered.at[index, "end_time"] = round(min(start + duration, 24.0), 4)
            cursor = start + duration

    return ordered.sort_index()


def _home_zones(persons: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Extract a person -> home zone lookup from synthetic population columns."""
    if persons is None or "person_id" not in persons.columns:
        return {}

    for column in ("hometaz", "home_taz", "HOMETAZ", "home_zone", "zone_id"):
        if column in persons.columns:
            values = pd.to_numeric(persons[column], errors="coerce")
            lookup = {
                str(pid): int(zone)
                for pid, zone in zip(persons["person_id"].astype(str), values.fillna(1))
            }
            return lookup
    return {}


def build_trip_chains(
    activities: pd.DataFrame,
    persons: Optional[pd.DataFrame] = None,
    mode: str = "home_based",
) -> pd.DataFrame:
    """
    Assemble activity episodes into a trip list.

    ``mode='home_based'`` treats each activity as one home-based tour (the
    tour-level structure used by the destination and mode models): a home ->
    activity leg and a matching activity -> home leg. Destination and mode
    predictions are attached by the pipeline after this step.
    """
    columns = [
        "tour_id", "trip_id", "person_id", "leg", "seq", "purpose",
        "OTAZ", "DTAZ", "home_zone", "depart_time", "arrive_time",
    ]
    if activities is None or len(activities) == 0:
        return pd.DataFrame(columns=columns)

    required = {"person_id", "activity_type", "start_time", "duration"}
    missing = required.difference(activities.columns)
    if missing:
        raise ValueError(f"activity schedule missing columns: {sorted(missing)}")

    homes = _home_zones(persons)
    trips: List[Dict[str, Any]] = []

    ordered = activities.sort_values(["person_id", "start_time"], kind="stable")
    for person_id, group in ordered.groupby("person_id", sort=False):
        pid = str(person_id)
        home_zone = int(group["home_zone"].iloc[0]) if "home_zone" in group.columns else homes.get(pid, 1)
        for tour_number, (_, activity) in enumerate(group.iterrows(), start=1):
            purpose = normalise_purpose(activity["activity_type"])
            start_minutes = int(round(float(activity["start_time"]) * 60.0))
            end_minutes = int(
                round(min(float(activity["start_time"]) + float(activity["duration"]), 24.0) * 60.0)
            )
            tour_id = f"{pid}_T{tour_number}"

            trips.append(
                {
                    "tour_id": tour_id,
                    "trip_id": f"{tour_id}_1",
                    "person_id": pid,
                    "leg": "outbound",
                    "seq": 1,
                    "purpose": purpose,
                    "OTAZ": home_zone,
                    "DTAZ": home_zone,
                    "home_zone": home_zone,
                    "depart_time": start_minutes,
                    "arrive_time": start_minutes,
                }
            )
            trips.append(
                {
                    "tour_id": tour_id,
                    "trip_id": f"{tour_id}_2",
                    "person_id": pid,
                    "leg": "return",
                    "seq": 2,
                    "purpose": "home",
                    "OTAZ": home_zone,
                    "DTAZ": home_zone,
                    "home_zone": home_zone,
                    "depart_time": end_minutes,
                    "arrive_time": end_minutes,
                }
            )

    return pd.DataFrame(trips, columns=columns)


def create_cdap_model(config: Optional[Dict] = None, seed: int = 42) -> CDAPModel:
    """Create a CDAP model from a configuration dictionary."""
    return CDAPModel(config, seed=seed)


def create_purpose_timing_model(
    config: Optional[Dict] = None, seed: int = 42
) -> PurposeTimingModel:
    """Create an empirical purpose timing model."""
    return PurposeTimingModel(config, seed=seed)
