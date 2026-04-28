import numpy as np

def monte_sample(probs):
    
    n_cases, n_alts = probs.shape
    all_nan = np.isnan(probs).all(axis=1)

    # replace those rows with equal probabilities
    probs[all_nan] = 1.0 / n_alts

    random_draws = np.random.rand(n_cases)

    # cumulative sums for sampling
    cdf = np.cumsum(probs, axis=1)

    # each sample is the FIRST alternative where CDF >= random draw
    sampled_alt = (cdf >= random_draws[:,None]).argmax(axis=1)

    # if your alternatives are labeled 1..N:
    predicted = sampled_alt
    return predicted


def monte_sample_key(probs, k=10):
    n_cases, n_alts = probs.shape

    # Handle rows that are all NaN
    all_nan = np.isnan(probs).all(axis=1)
    probs = probs.copy()
    probs[all_nan] = 1.0 / n_alts

    # Replace NaNs with 0 so they don't appear in top-k
    probs = np.nan_to_num(probs, nan=0.0)

    # Indices of top-k probabilities per row
    topk_idx = np.argpartition(probs, -k, axis=1)[:, -k:]

    # Mask: keep only top-k
    mask = np.zeros_like(probs, dtype=bool)
    mask[np.arange(n_cases)[:, None], topk_idx] = True

    # Zero out everything except top-k
    probs_topk = np.where(mask, probs, 0.0)

    # Renormalize so each row sums to 1
    row_sums = probs_topk.sum(axis=1, keepdims=True)
    probs_topk = probs_topk / row_sums

    # Monte Carlo draw
    random_draws = np.random.rand(n_cases)
    cdf = np.cumsum(probs_topk, axis=1)
    sampled_alt = (cdf >= random_draws[:, None]).argmax(axis=1)

    return sampled_alt


def add_time_dummies(df, start_col="start_of_day", end_col="end_of_day"):
    """
    Add binary time-of-day dummy variables for departure (start) and arrival (end) times.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing start and end columns.
    start_col : str
        Column name for start time (integer hours).
    end_col : str
        Column name for end time (integer hours).

    Returns
    -------
    pd.DataFrame
        Original dataframe with binary (0/1) dummy columns added.
    """

    # Copy so we don't overwrite
    df = df.copy()

    # --- Departure (Start) Dummies ---
    df["util_early_start_at_5"]             = (df[start_col] < 6).astype(int)
    df["util_am_peak_start_at_6"]           = (df[start_col] == 6).astype(int)
    df["util_am_peak_start_at_7"]           = (df[start_col] == 7).astype(int)
    df["util_am_peak_start_at_8"]           = (df[start_col] == 8).astype(int)
    df["util_am_peak_start_at_9"]           = (df[start_col] == 9).astype(int)
    df["util_midday_start_at_10_11_12"]     = ((df[start_col] > 9) & (df[start_col] < 13)).astype(int)
    df["util_midday_start_at_13_14_15"]     = ((df[start_col] > 12) & (df[start_col] < 16)).astype(int)
    df["util_pm_peak_start_at_16_17_18"]    = ((df[start_col] > 15) & (df[start_col] < 19)).astype(int)
    df["util_evening_start_at_19_20_21"]    = ((df[start_col] > 18) & (df[start_col] < 22)).astype(int)
    df["util_late_start_at_22_23"]          = (df[start_col] > 21).astype(int)

    # --- Arrival (End) Dummies ---
    df["util_early_end_at_5_6"]             = (df[end_col] < 7).astype(int)
    df["util_am_peak_end"]                  = ((df[end_col] > 6) & (df[end_col] < 10)).astype(int)
    df["util_midday_end_at_10_11_12"]       = ((df[end_col] > 9) & (df[end_col] < 13)).astype(int)
    df["util_midday_end_at_13_14"]          = ((df[end_col] > 12) & (df[end_col] < 15)).astype(int)
    df["util_pm_peak_end_at_15"]            = (df[end_col] == 15).astype(int)
    df["util_pm_peak_end_at_16"]            = (df[end_col] == 16).astype(int)
    df["util_pm_peak_end_at_17"]            = (df[end_col] == 17).astype(int)
    df["util_pm_peak_end_at_18"]            = (df[end_col] == 18).astype(int)
    df["util_evening_end_at_19_20_21"]      = ((df[end_col] > 18) & (df[end_col] < 22)).astype(int)
    df["util_late_end_at_22_23"]            = (df[end_col] > 21).astype(int)
    
    
    df['early_start']= (df[start_col] <6).astype(int)
    df['am_peak_start']= ((df[start_col] >=6) & (df[start_col] <=9)).astype(int)
    df['midday_start']= ((df[start_col] >=10) & (df[start_col] <=15)).astype(int)
    df['pm_peak_start']= ((df[start_col] >=16) & (df[start_col] <=18)).astype(int)
    df['evening_start']= ((df[start_col] >=19) & (df[start_col] <=21)).astype(int)
    df['late_start']= (df[start_col] >=22).astype(int)  
    df['early_end']= (df[end_col] <6).astype(int)
    df['am_peak_end']= ((df[end_col] >=6) & (df[end_col] <=9)).astype(int)
    df['midday_end']= ((df[end_col] >=10) & (df[end_col] <=15)).astype(int)
    df['pm_peak_end']= ((df[end_col] >=16) & (df[end_col] <=18)).astype(int)
    df['evening_end']= ((df[end_col] >=19) & (df[end_col] <=21)).astype(int)
    df['late_end']= (df[end_col] >=22).astype(int)
            
    

    return df


def time_hour(time):
    '''returns the time of starting..'''
    return int(time/60) +4

def get_start_dummy_name(start):
    if start < 6: return "util_early_start_at_5"
    elif start == 6: return "util_am_peak_start_at_6"
    elif start == 7: return "util_am_peak_start_at_7"
    elif start == 8: return "util_am_peak_start_at_8"
    elif start == 9: return "util_am_peak_start_at_9"
    elif 10 <= start <= 12: return "util_midday_start_at_10_11_12"
    elif 13 <= start <= 15: return "util_midday_start_at_13_14_15"
    elif 16 <= start <= 18: return "util_pm_peak_start_at_16_17_18"
    elif 19 <= start <= 21: return "util_evening_start_at_19_20_21"
    elif start >= 22: return "util_late_start_at_22_23"
    else: return None

def get_period(time, start = 'start'):
    if time <6: return f"early_{start}"
    elif 6 <= time <=9: return f"am_peak_{start}"
    elif 10 <= time <=15: return f"midday_{start}"
    elif 16 <= time <=18: return f"pm_peak_{start}"
    elif 19 <= time <=21: return f"evening_{start}"
    elif time >=22: return f"late_{start}"
    else: return None


def get_end_dummy_name(end):
    if end < 7: return "util_early_end_at_5_6"
    elif 7 <= end <= 9: return "util_am_peak_end"
    elif 10 <= end <= 12: return "util_midday_end_at_10_11_12"
    elif 13 <= end <= 14: return "util_midday_end_at_13_14"
    elif end == 15: return "util_pm_peak_end_at_15"
    elif end == 16: return "util_pm_peak_end_at_16"
    elif end == 17: return "util_pm_peak_end_at_17"
    elif end == 18: return "util_pm_peak_end_at_18"
    elif 19 <= end <= 21: return "util_evening_end_at_19_20_21"
    elif end >= 22: return "util_late_end_at_22_23"
    else: return None