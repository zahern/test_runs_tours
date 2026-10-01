# EB-Adjusted Street Risk Analysis

## Summary Statistics

- **Total road segments analyzed:** 7,517
- **Segments with event data:** 7,148
- **Segments with control data:** 7,431

### Event vs Control (Raw Counts)

- Total event braking: 573,373
- Total event speeding: 174,165
- Total control braking: 1,166,255
- Total control speeding: 358,626

### EB-Adjusted Estimates

- Total EB event risk: 747,529.3
- Total EB control risk: 1,524,792.6
- Total EB uplift (event - control): -777,263.3
- Mean EB weight (event): 0.276
- Mean EB weight (control): 0.213

### Top 20 Segments by EB-Adjusted Event Risk

| Rank | Road OSM ID | Road Name | EB Event Risk | Event Trips | EB Uplift |
|------|-------------|-----------|---------------|-------------|----------|
| 1 | 922464068 | Fish Lane | 7964.0 | 7578 | -9964.0 |
| 2 | 29702566 | Colchester Street | 7622.7 | 11555 | -9711.0 |
| 3 | 8036229 | Albert Street | 7488.1 | 7878 | -9860.0 |
| 4 | 23632342 | Fish Lane | 6317.1 | 6465 | -7842.0 |
| 5 | 28682261 | Roma Street | 4667.5 | 12093 | -5443.9 |
| 6 | 568325106 | Grey Street | 4631.5 | 13083 | -5483.0 |
| 7 | 930495875 | Stanley Street | 4596.8 | 15263 | -5831.9 |
| 8 | 1201590401 | Duncan Street | 4501.1 | 4532 | -5540.0 |
| 9 | 361656462 |  | 4229.4 | 7200 | -2676.1 |
| 10 | 23616379 | Bowen Bridge Road | 4010.0 | 6039 | -4320.6 |
| 11 | 281832758 | North Quay | 3584.8 | 9776 | -4334.9 |
| 12 | 1322803478 |  | 3560.3 | 3784 | -4604.1 |
| 13 | 530167436 | Adelaide Street | 3412.1 | 11665 | -4202.0 |
| 14 | 141292673 | Burnett Lane | 3264.2 | 4774 | -4190.1 |
| 15 | 1376779235 | Central Drive | 3233.2 | 3377 | -3553.0 |
| 16 | 785480299 |  | 3017.5 | 5574 | -1981.0 |
| 17 | 1132759032 | Stanley Street | 2972.5 | 5754 | -3599.8 |
| 18 | 456474243 |  | 2933.6 | 6982 | -3533.1 |
| 19 | 530170061 | Riverside Expressway | 2810.8 | 11778 | -3571.7 |
| 20 | 1132759033 | Stanley Street | 2795.2 | 6618 | -3362.0 |

### Interpretation

The EB-adjusted risk scores account for exposure (trip counts) and regression-to-the-mean bias. Segments with low exposure receive more weight on the model prediction, while high-exposure segments rely more on observed counts. The EB uplift (event - control) identifies segments where risk is specifically elevated on event days relative to matched non-event days.

**For the Police paper:** Use `eb_event` for overall event-day deployment ranking, and `eb_uplift` to identify event-specific corridors that warrant incremental staffing.
