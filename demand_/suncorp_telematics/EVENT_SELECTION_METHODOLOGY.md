# Suncorp Stadium Event Selection Methodology

## **Study Period: March 2026 (Telematics Data Window)**

### **Critical Constraint: Data Availability**
The telematics data available for analysis covers **2026-03-01 to 2026-04-01 ONLY** (one month). This is the exclusive window where connected-vehicle data was collected and provided.

---

## **Step 1: Identify ALL Fixtures in Data Window**

### **Search Criteria:**
- **Time period:** March 2026 (2026-03-01 to 2026-03-31)
- **Venue:** Suncorp Stadium, Brisbane
- **Home games only** (to study spectator arrival/dispersal patterns)

### **Initial Pool: March 2026 Suncorp Home Fixtures**

From the fixtures database (`fixtures_all.csv`), the following Brisbane Broncos home games occurred at Suncorp Stadium during the March 2026 data window:

| Event ID | Date | Kickoff | Sport | Round | Opponent | Result | Crowd | Day |
|----------|------|---------|-------|-------|----------|--------|-------|-----|
| **1** | Fri 6 Mar | 19:00 | NRL | Round 1 | Penrith Panthers | L 0–26 | 45,566 | Fri |
| **2** | Thu 12 Mar | 19:00 | NRL | Round 2 | Parramatta Eels | L 32–40 | 39,015 | Thu |
| **3** | Fri 27 Mar | 19:00 | NRL | Round 4 | Dolphins | W 26–12 | 45,882 | Fri |

**Total NRL fixtures in March 2026: 3**

---

## **Step 2: Expand to Multi-Sport Analysis**

To capture **sport-specific effects** and **crowd-size variation**, the search was expanded beyond just Broncos NRL matches to include **ALL major sporting codes** using Suncorp Stadium in March 2026.

### **Additional Sports Search (March 2026, Suncorp):**

Sources consulted:
- **Super Rugby (Queensland Reds):** Wikipedia Super Rugby fixtures 2026
- **A-League (Brisbane Roar):** Austadiums.com A-League 2025-26
- **State of Origin, Internationals:** QRL/ARL calendars

### **Additional Fixtures Found:**

| Event ID | Date | Kickoff | Sport | Match | Crowd | Day |
|----------|------|---------|-------|-------|-------|-----|
| **4** | Sat 14 Mar | 18:35 | Super Rugby | Reds vs Waratahs | W 26–17 | 18,087 | Sat |
| **5** | Sun 15 Mar | 17:15 | NRL | Titans vs Dolphins (Dolphins "home") | W 18–14 | 16,073 | Sun |
| **6** | Sat 21 Mar | 14:00 | A-League | Roar vs Wellington Phoenix | L 1–2 | 4,826 | Sat |

**Notes:**
- Event 4 (Super Rugby) is the **Queensland Reds home fixture** at Suncorp
- Event 5 (NRL) is a **Dolphins nominal "home" game** hosted at Suncorp (Dolphins use Suncorp for large-crowd matches)
- Event 6 (A-League) is the **Brisbane Roar home match** at Suncorp

---

## **Final Event List: 6 Fixtures Analyzed**

| ID | Date | Time | Sport | Home Team | Opponent | Crowd | Category |
|----|------|------|-------|-----------|----------|-------|----------|
| 1 | Fri 6 Mar | 19:00 | **NRL** | Broncos | Penrith | 45,566 | Large NRL |
| 2 | Thu 12 Mar | 19:00 | **NRL** | Broncos | Parramatta | 39,015 | Large NRL |
| 3 | Sat 14 Mar | 18:35 | **Super Rugby** | Reds | Waratahs | 18,087 | Medium rugby |
| 4 | Sun 15 Mar | 17:15 | **NRL** | Dolphins | Titans | 16,073 | Medium NRL |
| 5 | Sat 21 Mar | 14:00 | **A-League** | Roar | Wellington | 4,826 | Small football |
| 6 | Fri 27 Mar | 19:00 | **NRL** | Broncos | Dolphins | 45,882 | Large NRL |

---

## **Why These 6 Events?**

### **1. Data Window Constraint (Non-Negotiable)**
Only March 2026 telematics data exists. This is **not a sample**—it is the **entire available dataset**.

### **2. Sport Diversity (Analytical Benefit)**
- **4 NRL matches** (Rugby League: dominant winter code in Brisbane)
- **1 Super Rugby match** (Rugby Union: distinct crowd demographic)
- **1 A-League match** (Football/soccer: different crowd size, different day/time)

This diversity enables testing whether:
- **Sport type** affects driving behavior (rugby vs football spectators)
- **Kickoff time** matters (14:00 vs 17:15 vs 18:35 vs 19:00)
- **Day of week** matters (Thu/Fri/Sat/Sun)

### **3. Crowd Size Gradient (Critical for Analysis)**
Events span a **10:1 crowd ratio**:
- **Large (40k+):** Events 1, 2, 6
- **Medium (15-20k):** Events 3, 4
- **Small (<5k):** Event 5

This gradient tests whether **crowd size scales risky driving**, which is essential for the deployment model.

### **4. Home Team Representation**
- **Broncos (NRL):** 3 matches (primary tenant)
- **Dolphins (NRL):** 1 match (use Suncorp for large games)
- **Reds (Super Rugby):** 1 match (regular Suncorp tenant)
- **Roar (A-League):** 1 match (occasional Suncorp tenant)

This captures the **actual utilization pattern** of the stadium.

---

## **Events EXCLUDED (and Why)**

### **March 2026 Away Games (Excluded)**
- **Fri 20 Mar:** Broncos vs Melbourne at AAMI Park (Melbourne)
  - **Why excluded:** Not at Suncorp, no local traffic impact

### **Adjacent Months (Excluded — No Data)**
- **Thu 19 Feb:** Broncos vs Hull KR at MKM Stadium (World Club Challenge, UK)
- **Sat 4 Apr:** Broncos vs Gold Coast at Cbus Super Stadium
- **Fri 10 Apr:** Broncos vs North Queensland at Suncorp (45,582 crowd)
  - **Why excluded:** Telematics data ends 2026-03-31. **No data = cannot analyze.**

---

## **Matched Control Day Selection**

For each of the 6 events, **2 control days** are selected:

### **Control Day Criteria:**
1. **Same day-of-week** (Fri event → control Fri)
2. **Same clock window** (19:00 kickoff → 16:00–23:00 window)
3. **Non-event week** in March 2026
4. **No other Suncorp fixtures** on control day
5. **No public holidays**
6. **Nearest weeks** (±1 week, ±2 weeks) to minimize weather/seasonal drift

### **Control Pool (Example for Event 1: Fri 6 Mar 19:00)**
- **Candidate controls:** Fri 13 Mar, Fri 20 Mar, Fri 27 Mar
- **Exclusions:**
  - Fri 27 Mar = Event 6 (NRL Broncos vs Dolphins) → **EXCLUDED**
- **Final controls:** Fri 13 Mar, Fri 20 Mar (2 controls per event)

Result: **6 events × ~2 controls = ~12 control days → 18 event-control pairs analyzed**

---

## **Event Window Definition**

Each event is analyzed in a **7-hour observation window** centered on kickoff:

### **Window Phases:**
- **Pre-arrival:** Kickoff −3h to −0.5h (150 min)
  - Captures spectator arrival surge
- **During:** Kickoff −0.5h to +1.5h (120 min)
  - Seated period (NRL ~90 min, Super Rugby ~100 min, A-League ~105 min)
- **Post-dispersal:** Kickoff +1.5h to +4h (150 min)
  - Captures stadium exit and dispersal tail

**Total window:** 7 hours (420 minutes) per event

**Why fixed windows?**
- NRL (90 min), Super Rugby (80 min), A-League (90 min) have different durations
- Fixed windows enable **direct comparison** across sports
- Sensitivity analysis (in reproducibility bundle) tests window boundary effects

---

## **Summary: Event Selection Decision Tree**

```
Step 1: Filter by data availability
        └─> March 2026 only (telematics data constraint)

Step 2: Filter by venue
        └─> Suncorp Stadium only (study area)

Step 3: Filter by home games
        └─> Exclude away games (no local arrival/dispersal)

Step 4: Include all sports using venue
        └─> NRL (4), Super Rugby (1), A-League (1)

Result: 6 events (all fixtures meeting criteria)
```

**This is not a sample—it is the census of Suncorp events in the March 2026 data window.**

---

## **Validation: Why NOT 10 Sports?**

The paper text states **"six Suncorp events in the data window"** (Table 2 in main.tex lists 6 events).

**There are no "10 sports"—there are 6 events across 3 sports:**
1. **NRL** (4 events)
2. **Super Rugby** (1 event)
3. **A-League** (1 event)

If the todo list mentioned "10 sports," this may refer to:
- A misunderstanding OR
- A different analysis (e.g., 10 road segments, 10 time periods, etc.)

The **Suncorp stadium deployment paper analyzes 6 events, 3 sports, March 2026.**

---

## **Key Takeaway for Readers**

> **Why these 6 events?**  
> Because they are the **only major sporting events at Suncorp Stadium during the March 2026 telematics data window**. This is not a selective sample—it is the complete set of available fixtures within the data constraint.
