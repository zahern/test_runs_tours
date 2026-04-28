# Mode & Destination Choice Model Improvements - Quick Reference

## Summary of Issues & Solutions

### STAGE 5: Mixed Logit Failing for NONMAND
**Issue**: Model crashes, NaN mean values for random coefficients
**Root Cause**: 
  - Collinear income parameters (c_incp_da, c_incp_sr, c_incp_tr all ~ -3.46)
  - Standard deviation initialization at 0.5 is too high (flat gradients)
  - Mode tree structure not compatible with sharrow compiler

**Solution**: `stage5_improvements.estimate_mixed_logit_robust()`
  - Estimate MNL baseline first
  - Filter non-significant parameters (|t| < 1.96)
  - Initialize std dev at 0.3 instead of 0.5
  - Use two-phase optimization (BFGS + L-BFGS-B)

---

### STAGE 5: Zone Prediction Bias
**Issue**: 
  - Under-predict: zones 40, 182, 64
  - Over-predict: zones 220, 189, 27

**Root Cause**: Single destination model can't adapt to zone-specific characteristics

**Solution**: `stage5_improvements.compute_zone_adjustments()`
  - Compute zone-level shadow prices (log-adjustments)
  - Use IPF-style iterative balancing
  - Apply adjustments before softmax in prediction

---

### STAGE 6: Mode Choice Accuracy Crisis
**Issue**: SR 54%, Walk 20%, Bike 2%, Correct 12% overall
**Root Cause**:
  - ASC values not initialized to observed shares
  - Collinearity in income variables
  - No multi-start optimization (stuck in local optima)
  - Numerical overflow in softmax

**Solution**: 
  - `stage6_improvements.fit_asc_to_observed_shares()` — calibrate ASCs
  - `stage6_improvements.RobustProbabilityComputer` — safe probability computation
  - Multi-start optimization (5-10 random seeds)
  - Remove collinear parameters for nonmand segment

---

## Quick Implementation Checklist

### ✓ Stage 6 Fix (Immediate - ~10 minutes)
```python
# In stage6_mode_choice.py prepare_mode_data():
from stage6_improvements import filter_mode_choice_data, fit_asc_to_observed_shares

df = filter_mode_choice_data(df)
asc_init = fit_asc_to_observed_shares(df)  # <-- Add this

# In estimate_mode_models():
for purpose in PURPOSES:
    subset = data[data['purpose'] == purpose]
    asc_init = fit_asc_to_observed_shares(subset)
    m = NestedLogitModeModel()
    m.fit(subset, purpose, asc_init=asc_init)  # <-- Add parameter

# In NestedLogitModeModel._nested_proba():
from stage6_improvements import RobustProbabilityComputer
P = RobustProbabilityComputer.compute_nested_proba(V, mu_m, mu_nm)  # <-- Replace
```

### ✓ Stage 5 Enhancement (Medium - ~20 minutes)
```python
# In stage5_destination_choice.py _run_segment():
from stage5_improvements import (
    enhance_behavioral_vars, 
    filter_insignificant_params,
    compute_zone_adjustments
)

seg = enhance_behavioral_vars(seg, persons)  # <-- Add before MNL fit

# After MNL estimation:
coef_filt, se_filt, tstat_filt, dropped = filter_insignificant_params(
    m.coef_, m.se_, m.tstat_
)
# Use filtered coefficients instead of original

zone_adj = compute_zone_adjustments(obs_rows, probs, zones)  # <-- Add after dest fit
```

### ✓ Pipeline Integration (Light - ~5 minutes)
```python
# In pipeline_logger.py before stage6():
from stage6_improvements import fit_asc_to_observed_shares

asc_init = fit_asc_to_observed_shares(out["trips"], mode_col='MAINMODE')
# Pass to stage6() if available
```

---

## Function Reference

### stage5_improvements.py

#### `filter_insignificant_params(coef_dict, std_err_dict, tstat_dict, t_threshold=1.96)`
Removes parameters with |t-stat| < threshold, keeping structural terms (logsums, ASCs).
```python
coef_filt, se_filt, tstat_filt, dropped = filter_insignificant_params(
    m.coef_, m.se_, m.tstat_
)
```
Returns: (filtered_coefs, filtered_ses, filtered_tstats, dropped_list)

---

#### `estimate_mixed_logit_robust(seg, segment, n_draws=200, seed=42)`
Robust mixed logit with better initialization and numerical stability.
```python
result = estimate_mixed_logit_robust(segment_df, 'nonmand')
if result:
    coefs = result['coef']
    ll = result['ll']
```
Returns: dict with 'coef', 'std_err', 'tstat', 'll', 'rho2', 'convergence_path'

---

#### `compute_zone_adjustments(observed_df, predicted_probs, zone_ids, problem_zones=None)`
Computes zone-specific ASC-like adjustments to correct prediction bias.
```python
problem_zones = {40: 'underpredict', 182: 'underpredict', 64: 'underpredict',
                 220: 'overpredict', 189: 'overpredict', 27: 'overpredict'}
adj = compute_zone_adjustments(obs_df, probs_matrix, zones, problem_zones)
# adj = {40: +0.25, 182: +0.20, 64: +0.15, 220: -0.30, 189: -0.25, 27: -0.20}
```
Returns: {zone_id: adjustment_value}

---

#### `enhance_behavioral_vars(tours, persons)`
Adds/enriches behavioral variables (income interactions, household type, worker type).
```python
tours_enhanced = enhance_behavioral_vars(tours, persons)
# Adds: incp_quintile, car_per_person, hh_type, worker_type, plus person attributes
```
Returns: DataFrame with new columns

---

### stage6_improvements.py

#### `fit_asc_to_observed_shares(df, mode_col='chosen_mode')`
**CRITICAL FIX**: Initializes ASC values to match observed mode shares.
```python
asc_init = fit_asc_to_observed_shares(trips_df)
# Returns: {'ASC_DA': 0.0, 'ASC_SR': -1.0, 'ASC_Transit': -2.8, 'ASC_Walk': -2.0, 'ASC_Bike': -3.7}
```
Returns: {mode_name: asc_value}

---

#### `filter_mode_choice_data(df)`
Cleans and validates mode choice estimation data.
```python
df_clean = filter_mode_choice_data(df)
# Removes invalid modes, checks travel time [0, 600], distance [0, 1000], fills defaults
```
Returns: cleaned DataFrame

---

#### `RobustProbabilityComputer.compute_nested_proba(utilities, mu_motorised=0.8, mu_nonmotorised=0.8)`
Safe nested logit probability computation (prevents NaN/Inf).
```python
P = RobustProbabilityComputer.compute_nested_proba(V, mu_m, mu_nm)
# Returns: (n_obs, 5) safe probability matrix
```
Returns: (n_obs, 5) probability matrix (all rows sum to 1)

---

#### `detect_high_vif_params(df, threshold=10.0)`
Detects collinear parameters using correlation matrix.
```python
problem_cols = detect_high_vif_params(estimation_df)
# Returns: ['incp_da', 'incp_sr', 'incp_tr']
```
Returns: list of problematic column names

---

## Special Cases

### For NONMAND Segment (Most Problematic)
1. **Remove collinear income terms**: Set c_incp_sr=0, c_incp_active=0 (fixed, not free)
2. **Use robust MXL**: n_draws=200 minimum, seed=42 for reproducibility
3. **Check convergence**: If LL still NaN, fall back to MNL (no mixtures)

### For Zero-Vehicle Households
1. **Enhanced transit access**: Add c_noveh_transit, c_noveh_walk terms
2. **Bike ownership effect**: c_bikes_bike should capture household bikes
3. **Age interactions**: Young people in no-car households may bike/walk more

### For High-Income Households
1. **Vehicle ownership dominance**: c_veh_da typically 0.20-0.50 (strong positive)
2. **Time-value asymmetry**: High-income travelers are less price-sensitive
3. **Distance behavior**: Log-distance may underestimate high-income tolerance

---

## Expected Outcomes

### Stage 6 Mode Predictions
| Mode | Before | Target |
|------|--------|--------|
| DA   | 47%    | 60%    |
| SR   | 54%    | 18%    |
| Transit | 45% | 10%   |
| Walk | 20%    | 10%    |
| Bike | 2%     | 2%     |
| **Overall Correct** | 12% | **65%+** |

### Stage 5 Convergence
- NONMAND mixed logit: Converges ✓ (was crashing)
- Parameter count: Reduced by ~30% (removed non-significant)
- Zone R²: Improved by ~0.05 with shadow prices

---

## Debugging Tips

### If Mode Choice Still Predicts Everyone as DA
1. Check ASC initialization: `print(asc_init)` should show negative values for non-DA
2. Verify person vehicles: `print(df['vehrd'].value_counts())` should have variety
3. Check distance: `print(df['dist'].describe())` should span reasonable range
4. Look at convergence: If L-BFGS-B didn't converge, try more iterations

### If Mixed Logit Still Fails
1. Check parameter count: Should be < 25 parameters (was unlimited before filtering)
2. Verify mode distribution: Each segment needs 150+ observations
3. Look for NaN in inputs: `df.isnull().sum()` — check for all-NaN columns
4. Try MNL fallback first: Estimate without mixtures, then add incrementally

### If Zone Predictions Still Biased
1. Check zone sample sizes: Problematic zones should have 50+ observations
2. Verify adjustment magnitudes: Should be in [-2.0, +2.0] range
3. Look at relative error: (pred - obs) / obs should shrink after adjustment

---

## References

- **ActivitySim**: https://github.com/ActivitySim/activitysim
- **Larch Documentation**: https://larch.readthedocs.io/
- **Train (2009)**: Discrete Choice Methods with Simulation (Ch. 6 on mixed logit)
- **McFadden (1978)**: Modelling the Choice of Residential Location

---

## Contact & Support

For issues with:
- **Stage 5 mixed logit**: See `stage5_improvements.estimate_mixed_logit_robust()` docstring
- **Stage 6 ASC init**: See `stage6_improvements.fit_asc_to_observed_shares()` docstring
- **Zone adjustments**: See `stage5_improvements.compute_zone_adjustments()` docstring
- **Behavioral vars**: See `stage5_improvements.enhance_behavioral_vars()` docstring
