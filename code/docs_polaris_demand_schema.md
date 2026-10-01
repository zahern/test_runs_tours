# POLARIS Demand Database — Table Structures

Source: [POLARIS Studio docs, "Demand" module](https://polaris.taps.anl.gov/polaris-studio/demand/index.html), stable build (polaris 26.03 documentation).
Demand data is stored in an SQLite database named `ModelName-Demand.sqlite`. All pages loaded successfully; only the originally requested `demand/datamodel.html` URL 404'd — the data model page actually lives at `demand/auto_generated/datamodel.html` (content captured from there).

Conventions from the data model page:
- Geometry field (where present) is always called `geo`.
- Projection is always planar and in meters.
- Tables required by the data model have a capital first letter (e.g. `Household`, `Person`, `Trip`).
- Times are in seconds of simulation time, distances in meters.
- The demand database is generated/populated by the POLARIS simulation; the user's main interaction is extracting data from it.

## Demand Data Model Overview (how tables relate)

The demand database is an agent-based activity-travel output store. The core hierarchy is:

**Household → Person → Plan → Trip → Activity**

- **Household** is the top-level synthesized unit, attached to a network `Location` (supply table) via `Household.location`. It aggregates characteristics (persons, workers, vehicles, income, housing unit type, deliveries, etc.) either from ACS via population synthesis or carried over from a prior demand DB.
- **Person** (referenced by `Household.hhold` key field pattern) is each household member; each person has a **Plan** of **Trip**s connecting **Planned_Activity** / **Activity** stops. (Person/Plan/Planned_Activity field tables were not part of the requested set; relationships below are observed from FKs.)
- **Planned_Activity** — debug table of activities at planning time (only populated if `write_planned_activity_table = true`). FK to `Person(person)`; `trip` is always 0.
- **Activity** — output table of activities a person participates in; contains both not-yet-assigned planned activities (`trip == 0`) and assigned ones (`trip > 0`). FKs: `person` → `Person(person)`, plus `trip` → Trip table, `location_id`/`origin_id` → supply `Location` table.
- **Trip** — one row per person trip executed during simulation. FKs (declared in CREATE TABLE): `vehicle` → `Vehicle(vehicle_id)` (nullable; NULL for non-motorized trips), `person` → `Person(person)` (nullable). Also references the supply `Location` table via `origin`/`destination`, and Result.h5 path tables via `path`/`path_multimodal`. Only SOV-type trips count as person vehicle trips; freight trips also count as vehicle trips.
- **Vehicle** — all simulated vehicles: household-owned (`hhold` > 0, FK to `Household`; negative `hhold` = fleet-owned) and fleet-owned. FKs: `type` → `Vehicle_Type(type_id)`, `fleet` → `Fleet(fleet)` (nullable).
- **Fleet** — fleets operating in the simulation (TNC fleet, transit fleet, or freight-related fleet). Referenced by `Vehicle.fleet`.
- **Mode** — dictionary table providing a text representation of the hard-coded `Vehicle_Type_Keys` enum; used by `Trip.mode` and the `mode`/`type` TEXT fields on `Activity`/`Planned_Activity`.
- **Activity_Type** — codes activity types to their text description.

Cross-module references: several demand tables FK to the supply **Location** table (`location_id`, `origin_id`, `origin`, `destination`), and Trips reference Result.h5 **path** entries. Enumeration values (mode codes, activity type codes, household types, trip types) are hard-coded C++ enums mirrored in the docs; the `Mode` and `Activity_Type` DB tables document them.

> Note on a doc inconsistency: `Trip.vehicle` is described as FK "Vehicle(vehicle)" in the field table, but its CREATE TABLE constraint says `REFERENCES "Vehicle" ("vehicle_id")` — the SQL is authoritative here (also `Trip.person` FK is `References "Person" ("person")`).

---

# Table: Activity_Type

> "THIS TABLE [CODES] ACTIVITY TYPE CODES TO THEIR TEXT DESCRIPTION"

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| activity_type_id * | INTEGER | NO | | | Activity type id |
| activity_type_description | TEXT | NO | | | Activity type description |

(\* = Primary key)

No enum tables on this page.

```sql
CREATE TABLE "Activity_Type" (
  "activity_type_id"          INTEGER NOT NULL PRIMARY KEY,
  "activity_type_description" TEXT NOT NULL)
```

---

# Table: Activity

> An output table for activities a person participates in. Note this table contains both planned activities which are not assigned to the network yet (`trip == 0`) as well as assigned activities (`trip > 0`).

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| id * | INTEGER | NO | | | Unique identifier of this activity |
| seq_num | INTEGER | NO | 0 | | Order in which activities were generated per person; note home activities are +=100 and split activities +=1000 |
| location_id | INTEGER | NO | 0 | | Location at which this activity took place (foreign key to the Location table) |
| start_time | REAL | YES | 0 | | Simulation time in seconds when this activity starts (units: seconds) |
| duration | REAL | YES | 0 | | Duration of the activity in seconds of simulation time (units: seconds) |
| mode | TEXT | NO | '' | | Mode to reach this activity, possible values are keys in Values at [mode] |
| type | TEXT | NO | '' | | Type of the activity, possible values are keys in Values at [type] |
| person | INTEGER | NO | | Person(person) | Person undertaking this activity (foreign key to the Person table) |
| trip | INTEGER | NO | | | The trip which was used to reach this activity (foreign key to trip table); 0 indicates this is a planned activity without trip |
| origin_id | INTEGER | NO | 0 | | The location of the previous activity (foreign key to the Location table) |

(\* = Primary key)

```sql
CREATE TABLE "Activity" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "seq_num" INTEGER NOT NULL DEFAULT 0,
  "location_id" INTEGER NOT NULL DEFAULT 0,
  "start_time" REAL NULL DEFAULT 0,
  "duration" REAL NULL DEFAULT 0,
  "mode" TEXT NOT NULL DEFAULT '',
  "type" TEXT NOT NULL DEFAULT '',
  "person" INTEGER NOT NULL,
  "trip" INTEGER NOT NULL,
  "origin_id" INTEGER NOT NULL DEFAULT 0,
  CONSTRAINT "person_fk"
    FOREIGN KEY ("person")
    REFERENCES "Person" ("person")
    DEFERRABLE INITIALLY DEFERRED)
```

## Enum: mode (Activity.mode)

| Value | Name | Description |
|---|---|---|
| 0 | SOV | Car driver |
| 1 | HOV_HH | Car passenger in household vehicle |
| 2 | HOV | Car passenger in household vehicle |
| 3 | TRUCK | |
| 4 | BUS | Mode identifier associated with an articulated 60 ft bus vehicle, if the [description truncated in source] |
| 5 | RAIL | Public transit with walk access and egress to any combination of rail sub-modes (subway, light [description truncated in source] |
| 7 | BICYCLE | Privately owned bicycle |
| 8 | WALK | Walking |
| 9 | TAXI | Taxi and TNC ride hailing for passengers |
| 10 | SCHOOLBUS | School bus |
| 11 | PARK_AND_RIDE | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 12 | KISS_AND_RIDE | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 13 | PARK_AND_RAIL | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 14 | KISS_AND_RAIL | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 15 | TNC_AND_RIDE | Public transit, with access and/or egress by taxi or rideshare. |
| 17 | MD_TRUCK | Medium Duty Trucks, Classes 3-6 (10,001-26,000 lbs) according to FHWA classification |
| 18 | HD_TRUCK | Heavy Duty Trucks, Class 7-8 (> 26,001 lbs) according to FHWA classification |
| 19 | BPLATE | Pickup trucks and vans (Chicago model only), Class 2 (< 10,000 lbs) according to FHWA [description truncated in source] |
| 20 | LD_TRUCK | Light Duty Trucks, Class 2 (< 10,000 lbs) according to FHWA classification |
| 22 | BUS40 | |
| 23 | BUS60 | |
| 25 | RIDE_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can use any any [description truncated in source] |
| 26 | RIDE_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can use [description truncated in source] |
| 27 | RAIL_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can only use rail modes [description truncated in source] |
| 28 | RAIL_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can only [description truncated in source] |
| 29 | MICROM | Docked micromobility mode (if micromobility and transit has bike time but no transit time) |
| 30 | MICROM_NODOCK | Dockless micromobility modes (if dockless micromobility and transit has bike time but [description truncated in source] |
| 31 | MICROM_AND_TRANSIT | Docked micromobility modes for access/egress to transit |
| 32 | MICROM_NODOCK_AND_TRANSIT | Dockless micromobility and transit modes combination |
| 33 | ODDELIVERY | Groceries and meals delivery |
| 999 | FAIL_MODE | If a traveler left home using drive-to-transit but returning home via transit and [description truncated in source] |
| 1000 | FAIL_ROUTE | Not used |
| 1001 | FAIL_REROUTE | If too many rerouting happens or the rerouting fails for a multimodal trip, the mode [description truncated in source] |
| 1002 | FAIL_UNPARK | If the traveler arrives to their parked vehicle on a return (transit-and-drive) trip [description truncated in source] |
| 1003 | FAIL_UNPARK2 | |
| 1004 | FAIL_MODE1 | |
| 1005 | FAIL_MODE2 | |
| 1006 | FAIL_MODE3 | |
| 1007 | FAIL_ROUTE_ACTIVE | If the multimodal routing fails for a walking or biking trip the mode becomes [description truncated in source] |
| 1008 | FAIL_ROUTE_WALK_AND_TRANSIT | |
| 1009 | FAIL_ROUTE_DRIVE_TO_TRANSIT | |
| 1010 | FAIL_ROUTE_DRIVE_FROM_TRANSIT | If the multimodal routing fails for a drive-from-transit trip; [description truncated in source] |
| 1011 | FAIL_ROUTE_TNC_AND_TRANSIT | |
| 1012 | FAIL_ROUTE_TNC | If the multimodal routing fails for a tnc trip; replan mode is triggered |
| 1013 | FAIL_ROUTE_SOV | Not used |
| 1014 | FAIL_ROUTE_MICROMOBILITY | |
| 1015 | NO_MOVE | If the origin and destination locations are the same, no movement needs to happen |
| 9999 | UNSIMULATED | If both the origin and destination locations fall into the no-simulation portion of [description truncated in source] |

## Enum: type (Activity.type)

| Value | Name | Description |
|---|---|---|
| -1 | DELETED | |
| -2 | EMPTY | |
| 0 | WORK_AT_HOME_ACTIVITY | |
| 1 | AT_HOME_ACTIVITY | |
| 2 | PRIMARY_WORK_ACTIVITY | |
| 3 | OTHER_WORK_ACTIVITY | |
| 4 | SCHOOL_ACTIVITY | |
| 5 | MAJOR_SHOPPING_ACTIVITY | |
| 6 | OTHER_SHOPPING_ACTIVITY | |
| 7 | SERVICE_VEHICLE_ACTIVITY | |
| 8 | HEALTHCARE_ACTIVITY | |
| 9 | PERSONAL_BUSINESS_ACTIVITY | |
| 10 | ERRANDS_ACTIVITY | |
| 11 | PICK_UP_OR_DROP_OFF_ACTIVITY | |
| 12 | EAT_OUT_ACTIVITY | |
| 13 | RELIGIOUS_OR_CIVIC_ACTIVITY | |
| 14 | LEISURE_ACTIVITY | |
| 15 | RECREATION_ACTIVITY | |
| 16 | SOCIAL_ACTIVITY | |
| 17 | CHANGE_TRANSPORTATION_ACTIVITY | |
| 18 | OTHER_ACTIVITY | |
| 22 | PART_TIME_WORK_ACTIVITY | |
| 23 | EV_CHARGING_ACTIVITY | |

---

# Table: Household

> Records the households synthesized in the simulation and all associated characteristics and metrics aggregated at the household level. Certain columns are directly obtained from the ACS, while others are synthesized through demand models executed during simulation. Records are either a result of population synthesis or are being moved from one Demand database to the next when reading population from database.

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| household * | INTEGER | NO | | | The unique identifier for the household |
| hhold | INTEGER | NO | | | The upstream (i.e. PUMS) identifier for the household |
| location | INTEGER | NO | | | The location to which this household is attached (foreign key to the Location table) |
| persons | INTEGER | NO | 0 | | Number of household members |
| workers | INTEGER | NO | 0 | | Number of workers in the household as defined by WiF from the ACS (number of people over 15 that worked one week or more in the last year, capped at 3+) |
| vehicles | INTEGER | NO | 0 | | Number of vehicles in the household |
| type | INTEGER | NO | 0 | | Household type as defined by the ACS, values at [type] enum |
| income | INTEGER | NO | 0 | | Annual household income (ACS field HINCP) |
| bikes | INTEGER | NO | 0 | | Number of bicycles in the household |
| housing_unit_type | INTEGER | NO | 0 | | Type of the housing unit, values at [housing_unit_type] enum |
| ecom | INTEGER | NO | 0 | | Number of e-commerce packages received in the simulated day by the household |
| delRat | REAL | YES | 0 | | Delivery ratio |
| dispose_veh | INTEGER | NO | | | boolean flag - were one or more vehicles disposed within the household? |
| time_in_home | REAL | NO | 0 | | Time this household has been resident at this location (unit: years) |
| Has_Residential_Charging | INTEGER | NO | 0 | | boolean flag - does household have an EV charging station? |
| num_groceries | INTEGER | NO | 0 | | Number of grocery deliveries received in the simulated day by the household |
| num_meals | INTEGER | NO | 0 | | Number of meal deliveries received in the simulation day by the household |

(\* = Primary key)

```sql
CREATE TABLE "Household" (
  "household" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hhold" INTEGER NOT NULL,
  "location" INTEGER NOT NULL,
  "persons" INTEGER NOT NULL DEFAULT 0,
  "workers" INTEGER NOT NULL DEFAULT 0,
  "vehicles" INTEGER NOT NULL DEFAULT 0,
  "type" INTEGER NOT NULL DEFAULT 0,
  "income" INTEGER NOT NULL DEFAULT 0,
  "bikes" INTEGER NOT NULL DEFAULT 0,
  "housing_unit_type" INTEGER NOT NULL DEFAULT 0,
  "ecom" INTEGER NOT NULL DEFAULT 0,
  "delRat" REAL NULL DEFAULT 0,
  "dispose_veh" INTEGER NOT NULL,
  "time_in_home" REAL NOT NULL DEFAULT 0,
  "Has_Residential_Charging" INTEGER NOT NULL DEFAULT 0,
  "num_groceries" INTEGER NOT NULL DEFAULT 0,
  "num_meals" INTEGER NOT NULL DEFAULT 0
  )
```

## Enum: type (Household.type)

| Value | Name | Description |
|---|---|---|
| 1 | HHT_OWN_FAMILY_MARRIED | |
| 2 | HHT_OWN_FAMILY_MALE | |
| 3 | HHT_OWN_FAMILY_FEMALE | |
| 4 | HHT_OWN_NONFAMILY_SINGLE | |
| 5 | HHT_OWN_NONFAMILY_OTHER | |
| 6 | HHT_RENT_FAMILY_MARRIED | |
| 7 | HHT_RENT_FAMILY_MALE | |
| 8 | HHT_RENT_FAMILY_FEMALE | |
| 9 | HHT_RENT_NONFAMILY_SINGLE | |
| 10 | HHT_RENT_NONFAMILY_OTHER | |
| 11 | HHT_GQ | |

## Enum: housing_unit_type (Household.housing_unit_type)

| Value | Name | Description |
|---|---|---|
| 1 | HU_SINGLE_FAMILY | |
| 2 | HU_SMALL_APT_BLD | |
| 3 | HU_LARGE_APT_BLD | |
| 4 | HU_OTHER | |
| 5 | HU_GROUP_QUARTERS | |

---

# Table: Trip

> All person trips executed during simulation in POLARIS is logged here as a separate record. Note that only the SOV-related record can be counted as a vehicle trip for person-related travel. All freight trips can also be counted as a vehicle trip.

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| trip_id * | INTEGER | NO | | | Unique identifier of this trip |
| hhold | INTEGER | NO | 0 | | Not used |
| path | INTEGER | NO | -1 | | Identifier of the corresponding entry in the path table (in Result.h5) for this path, will be -1 if trajectory info not written for this trip. |
| path_multimodal | INTEGER | NO | -1 | | Identifier of the corresponding entry in the multi-modal path table (in Result.h5) for this path, will be -1 if this is not a MM trip. |
| tour | INTEGER | NO | 0 | | Not used |
| trip | INTEGER | NO | 0 | | Not used |
| start | REAL | YES | 0 | | Simulation time when this trip starts (units: seconds) |
| end | REAL | YES | 0 | | Simulation time when this trip ends (units: seconds) |
| duration | REAL | YES | 0 | | Not used |
| experienced_gap | REAL | YES | 0 | | Gap experienced by person based on executed travel time (provided by end minus start) compared to routed travel time. |
| origin | INTEGER | NO | 0 | | The location where this trip begins (foreign key to the Location table) |
| destination | INTEGER | NO | 0 | | The location where this trip ends (foreign key to the Location table) |
| purpose | INTEGER | NO | 0 | | Currently used only to distinguish freight trips as E-Commerce or not |
| mode | INTEGER | NO | 0 | | The mode used to execute the trip. Values at [mode] enum. |
| constraint | INTEGER | NO | 0 | | Not used |
| priority | INTEGER | NO | 0 | | Not used |
| vehicle | INTEGER | YES | | Vehicle(vehicle) | Vehicle ID used to complete the travel. Can be NULL for non-motorized trips. |
| passengers | INTEGER | NO | 0 | | Not used |
| type | INTEGER | NO | 0 | | Trip type is used to differentiate trips that are synthetically generated versus those provided as an input. Further classification by source exists. Acceptable values at [type] enum. |
| partition | INTEGER | NO | 0 | | Not used |
| person | INTEGER | YES | | Person(person) | Person who was primarily involved in the trip (foreign key to the Person table) |
| travel_distance | REAL | YES | 0 | | Distance traveled in executing this trip (units: meters) |
| skim_travel_time | REAL | YES | 0 | | Travel time from the skim table for comparison with execution (units: seconds) |
| routed_travel_time | REAL | YES | 0 | | Travel time from the router for comparison with execution (units: seconds) |
| access_egress_ovtt | REAL | YES | 0 | | Out of vehicle travel time from non-motorized mode that is not the main mode of the trip (units: seconds) |
| toll | REAL | YES | 0 | | Toll paid in executing the trip (units: $USD) |
| has_artificial_trip | INTEGER | NO | 0 | | Integer to denote the type of artificial trip, values at [has_artificial_trip] enum |
| number_of_switches | INTEGER | NO | 0 | | Integer value indicating the number of times the path was changed to account for downstream congestion or when waiting too long at a link |
| request | INTEGER | NO | 0 | | Not used |
| monetary_cost | REAL | YES | 0 | | Dollar amount paid when making the trip (units: $USD) |
| initial_energy_level | REAL | YES | 0 | | The battery energy level when starting the trip (units: Watt Hours) |
| final_energy_level | REAL | YES | 0 | | The battery energy level when completing the trip (units: Watt Hours) |

(\* = Primary key)

```sql
CREATE TABLE "Trip" (
  "trip_id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hhold" INTEGER NOT NULL DEFAULT 0,
  "path" INTEGER NOT NULL DEFAULT -1,
  "path_multimodal" INTEGER NOT NULL DEFAULT -1,
  "tour" INTEGER NOT NULL DEFAULT 0,
  "trip" INTEGER NOT NULL DEFAULT 0,
  "start" REAL NULL DEFAULT 0,
  "end" REAL NULL DEFAULT 0,
  "duration" REAL NULL DEFAULT 0,
  "experienced_gap" REAL NULL DEFAULT 0,
  "origin" INTEGER NOT NULL DEFAULT 0,
  "destination" INTEGER NOT NULL DEFAULT 0,
  "purpose" INTEGER NOT NULL DEFAULT 0,
  "mode" INTEGER NOT NULL DEFAULT 0,
  "constraint" INTEGER NOT NULL DEFAULT 0,
  "priority" INTEGER NOT NULL DEFAULT 0,
  "vehicle" INTEGER NULL,
  "passengers" INTEGER NOT NULL DEFAULT 0,
  "type" INTEGER NOT NULL DEFAULT 0,
  "partition" INTEGER NOT NULL DEFAULT 0,
  "person" INTEGER NULL,
  "travel_distance" REAL NULL DEFAULT 0,
  "skim_travel_time" REAL NULL DEFAULT 0,
  "routed_travel_time" REAL NULL DEFAULT 0,
  "access_egress_ovtt" REAL NULL DEFAULT 0,
  "toll" REAL NULL DEFAULT 0,
  "has_artificial_trip" INTEGER NOT NULL DEFAULT 0,
  "number_of_switches" INTEGER NOT NULL DEFAULT 0,
  "request" INTEGER NOT NULL DEFAULT 0,
  "monetary_cost" REAL NULL DEFAULT 0,
  "initial_energy_level" REAL NULL DEFAULT 0,
  "final_energy_level" REAL NULL DEFAULT 0,
  CONSTRAINT "vehicle_fk"
    FOREIGN KEY ("vehicle")
    REFERENCES "Vehicle" ("vehicle_id")
    DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT "person_fk"
    FOREIGN KEY ("person")
    REFERENCES "Person" ("person")
    DEFERRABLE INITIALLY DEFERRED)
```

## Enum: mode (Trip.mode)

| Value | Name | Description |
|---|---|---|
| 0 | SOV | Car driver |
| 1 | HOV_HH | Car passenger in household vehicle |
| 2 | HOV | Car passenger in household vehicle |
| 3 | TRUCK | |
| 4 | BUS | Mode identifier associated with an articulated 60 ft bus vehicle, if the [description truncated in source] |
| 5 | RAIL | Public transit with walk access and egress to any combination of rail sub-modes (subway, light [description truncated in source] |
| 7 | BICYCLE | Privately owned bicycle |
| 8 | WALK | Walking |
| 9 | TAXI | Taxi and TNC ride hailing for passengers |
| 10 | SCHOOLBUS | School bus |
| 11 | PARK_AND_RIDE | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 12 | KISS_AND_RIDE | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 13 | PARK_AND_RAIL | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 14 | KISS_AND_RAIL | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 15 | TNC_AND_RIDE | Public transit, with access and/or egress by taxi or rideshare. |
| 17 | MD_TRUCK | Medium Duty Trucks, Classes 3-6 (10,001-26,000 lbs) according to FHWA classification |
| 18 | HD_TRUCK | Heavy Duty Trucks, Class 7-8 (> 26,001 lbs) according to FHWA classification |
| 19 | BPLATE | Pickup trucks and vans (Chicago model only), Class 2 (< 10,000 lbs) according to FHWA [description truncated in source] |
| 20 | LD_TRUCK | Light Duty Trucks, Class 2 (< 10,000 lbs) according to FHWA classification |
| 22 | BUS40 | |
| 23 | BUS60 | |
| 25 | RIDE_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can use any any [description truncated in source] |
| 26 | RIDE_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can use [description truncated in source] |
| 27 | RAIL_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can only use rail modes [description truncated in source] |
| 28 | RAIL_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can only [description truncated in source] |
| 29 | MICROM | Docked micromobility mode (if micromobility and transit has bike time but no transit time) |
| 30 | MICROM_NODOCK | Dockless micromobility modes (if dockless micromobility and transit has bike time but [description truncated in source] |
| 31 | MICROM_AND_TRANSIT | Docked micromobility modes for access/egress to transit |
| 32 | MICROM_NODOCK_AND_TRANSIT | Dockless micromobility and transit modes combination |
| 33 | ODDELIVERY | Groceries and meals delivery |
| 999 | FAIL_MODE | If a traveler left home using drive-to-transit but returning home via transit and [description truncated in source] |
| 1000 | FAIL_ROUTE | Not used |
| 1001 | FAIL_REROUTE | If too many rerouting happens or the rerouting fails for a multimodal trip, the mode [description truncated in source] |
| 1002 | FAIL_UNPARK | If the traveler arrives to their parked vehicle on a return (transit-and-drive) trip [description truncated in source] |
| 1003 | FAIL_UNPARK2 | |
| 1004 | FAIL_MODE1 | |
| 1005 | FAIL_MODE2 | |
| 1006 | FAIL_MODE3 | |
| 1007 | FAIL_ROUTE_ACTIVE | If the multimodal routing fails for a walking or biking trip the mode becomes [description truncated in source] |
| 1008 | FAIL_ROUTE_WALK_AND_TRANSIT | |
| 1009 | FAIL_ROUTE_DRIVE_TO_TRANSIT | |
| 1010 | FAIL_ROUTE_DRIVE_FROM_TRANSIT | If the multimodal routing fails for a drive-from-transit trip; [description truncated in source] |
| 1011 | FAIL_ROUTE_TNC_AND_TRANSIT | |
| 1012 | FAIL_ROUTE_TNC | If the multimodal routing fails for a tnc trip; replan mode is triggered |
| 1013 | FAIL_ROUTE_SOV | Not used |
| 1014 | FAIL_ROUTE_MICROMOBILITY | |
| 1015 | NO_MOVE | If the origin and destination locations are the same, no movement needs to happen |
| 9999 | UNSIMULATED | If both the origin and destination locations fall into the no-simulation portion of [description truncated in source] |

## Enum: type (Trip.type)

| Value | Name | Description |
|---|---|---|
| -1 | NULLTRIP | Invalid trip types |
| 11 | ABM | Trips which are synthesized by travelers during the simulation |
| 22 | EXTERNAL | Fixed demand read in from DB (background traffic, freight, out-of-model SOV, etc.) |
| 32 | TNC_VEHICLE | Movements undertaken by TNC vehicles, written to the TNC_Trip table not Trip |
| 33 | TNC_REQUEST | Fixed TNC Demand which is read in from the Trips table of database |
| 44 | FREIGHT | Trips synthetically generated by CRISTAL Freight model |
| 45 | FREIGHT_AV | Freight trips operated by automated truck carriers |
| 55 | TRANSIT | Movements undertaken by transit vehicles (operated by transit agencies) |
| 99 | UNSIMULATED | If both the origin and destination locations fall into the no-simulation portion of [description truncated in source] |

## Enum: has_artificial_trip (Trip.has_artificial_trip)

| Value | Name | Description |
|---|---|---|
| 0 | ALL_GOOD | |
| 1 | NOT_ROUTED | |
| 2 | CONGESTION_REMOVAL | |
| 3 | SIMULATION_ENDED | |
| 4 | STUCK_IN_ENTRY_QUEUE | |

---

# Table: Traveler

> THERE IS CURRENTLY NO DOCUMENTATION FOR THIS TABLE (per the source page)

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| traveler * | INTEGER | NO | | | |
| hhold | INTEGER | NO | 0 | | |
| person | INTEGER | NO | 0 | | |
| tour | INTEGER | NO | 0 | | |
| trip | INTEGER | NO | 0 | | |
| mode | INTEGER | NO | 0 | | |
| time | REAL | YES | 0 | | |
| distance | REAL | YES | 0 | | |
| speed | REAL | YES | 0 | | |
| link | INTEGER | NO | 0 | | |
| dir | INTEGER | NO | 0 | | |
| lane | INTEGER | NO | 0 | | |
| offset | REAL | YES | 0 | | |
| route | INTEGER | NO | 0 | | |

(\* = Primary key)

No enum tables on this page.

```sql
CREATE TABLE "Traveler" (
  "traveler" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hhold" INTEGER NOT NULL DEFAULT 0,
  "person" INTEGER NOT NULL DEFAULT 0,
  "tour" INTEGER NOT NULL DEFAULT 0,
  "trip" INTEGER NOT NULL DEFAULT 0,
  "mode" INTEGER NOT NULL DEFAULT 0,
  "time" REAL NULL DEFAULT 0,
  "distance" REAL NULL DEFAULT 0,
  "speed" REAL NULL DEFAULT 0,
  "link" INTEGER NOT NULL DEFAULT 0,
  "dir" INTEGER NOT NULL DEFAULT 0,
  "lane" INTEGER NOT NULL DEFAULT 0,
  "offset" REAL NULL DEFAULT 0,
  "route" INTEGER NOT NULL DEFAULT 0)
```

---

# Table: Planned_Activity

> An output table for planned activities a person wants to participate in. Note this is a debug table and records activity attributes at the time of planning in activity generation. As such it will only be populated if `write_planned_activity_table` is set to true.

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| id * | INTEGER | NO | | | Unique identifier of this activity |
| seq_num | INTEGER | NO | 0 | | Order in which activities were generated per person; note home activities are +=100 and split activities +=1000 |
| location_id | INTEGER | NO | 0 | | Location of the activity (foreign key to the Location table) |
| start_time | REAL | YES | 0 | | Start time of the activity at the time of planning (units: seconds) |
| duration | REAL | YES | 0 | | Duration of the activity at the time of planning (units: seconds) |
| mode | TEXT | NO | '' | | Mode to reach this activity at the time of planning, possible values are keys in Values at [mode] |
| type | TEXT | NO | '' | | Type of the activity, possible values are keys in Values at [type] |
| person | INTEGER | NO | | Person(person) | The person undertaking this activity (foreign key to the Person table) |
| trip | INTEGER | NO | | | Always 0 |
| origin_id | INTEGER | NO | 0 | | Location of previous activity at the time of planning (foreign key to the Location table) |
| status | INTEGER | NO | 0 | | Planning status. 0 means added to schedule, larger than zero indicates planning failure and activity is dropped |
| plan_time | INTEGER | NO | 0 | | Simulation time when activity is scheduled (units: seconds) |

(\* = Primary key)

```sql
CREATE TABLE "Planned_Activity" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "seq_num" INTEGER NOT NULL DEFAULT 0,
  "location_id" INTEGER NOT NULL DEFAULT 0,
  "start_time" REAL NULL DEFAULT 0,
  "duration" REAL NULL DEFAULT 0,
  "mode" TEXT NOT NULL DEFAULT '',
  "type" TEXT NOT NULL DEFAULT '',
  "person" INTEGER NOT NULL,
  "trip" INTEGER NOT NULL,
  "origin_id" INTEGER NOT NULL DEFAULT 0,
  "status" INTEGER NOT NULL DEFAULT 0,
  "plan_time" INTEGER NOT NULL DEFAULT 0,
  CONSTRAINT "person_fk"
    FOREIGN KEY ("person")
    REFERENCES "Person" ("person")
    DEFERRABLE INITIALLY DEFERRED)
```

## Enum: mode (Planned_Activity.mode)

| Value | Name | Description |
|---|---|---|
| 0 | SOV | Car driver |
| 1 | HOV_HH | Car passenger in household vehicle |
| 2 | HOV | Car passenger in household vehicle |
| 3 | TRUCK | |
| 4 | BUS | Mode identifier associated with an articulated 60 ft bus vehicle, if the [description truncated in source] |
| 5 | RAIL | Public transit with walk access and egress to any combination of rail sub-modes (subway, light [description truncated in source] |
| 7 | BICYCLE | Privately owned bicycle |
| 8 | WALK | Walking |
| 9 | TAXI | Taxi and TNC ride hailing for passengers |
| 10 | SCHOOLBUS | School bus |
| 11 | PARK_AND_RIDE | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 12 | KISS_AND_RIDE | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 13 | PARK_AND_RAIL | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 14 | KISS_AND_RAIL | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 15 | TNC_AND_RIDE | Public transit, with access and/or egress by taxi or rideshare. |
| 17 | MD_TRUCK | Medium Duty Trucks, Classes 3-6 (10,001-26,000 lbs) according to FHWA classification |
| 18 | HD_TRUCK | Heavy Duty Trucks, Class 7-8 (> 26,001 lbs) according to FHWA classification |
| 19 | BPLATE | Pickup trucks and vans (Chicago model only), Class 2 (< 10,000 lbs) according to FHWA [description truncated in source] |
| 20 | LD_TRUCK | Light Duty Trucks, Class 2 (< 10,000 lbs) according to FHWA classification |
| 22 | BUS40 | |
| 23 | BUS60 | |
| 25 | RIDE_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can use any any [description truncated in source] |
| 26 | RIDE_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can use [description truncated in source] |
| 27 | RAIL_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can only use rail modes [description truncated in source] |
| 28 | RAIL_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can only [description truncated in source] |
| 29 | MICROM | Docked micromobility mode (if micromobility and transit has bike time but no transit time) |
| 30 | MICROM_NODOCK | Dockless micromobility modes (if dockless micromobility and transit has bike time but [description truncated in source] |
| 31 | MICROM_AND_TRANSIT | Docked micromobility modes for access/egress to transit |
| 32 | MICROM_NODOCK_AND_TRANSIT | Dockless micromobility and transit modes combination |
| 33 | ODDELIVERY | Groceries and meals delivery |
| 999 | FAIL_MODE | If a traveler left home using drive-to-transit but returning home via transit and [description truncated in source] |
| 1000 | FAIL_ROUTE | Not used |
| 1001 | FAIL_REROUTE | If too many rerouting happens or the rerouting fails for a multimodal trip, the mode [description truncated in source] |
| 1002 | FAIL_UNPARK | If the traveler arrives to their parked vehicle on a return (transit-and-drive) trip [description truncated in source] |
| 1003 | FAIL_UNPARK2 | |
| 1004 | FAIL_MODE1 | |
| 1005 | FAIL_MODE2 | |
| 1006 | FAIL_MODE3 | |
| 1007 | FAIL_ROUTE_ACTIVE | If the multimodal routing fails for a walking or biking trip the mode becomes [description truncated in source] |
| 1008 | FAIL_ROUTE_WALK_AND_TRANSIT | |
| 1009 | FAIL_ROUTE_DRIVE_TO_TRANSIT | |
| 1010 | FAIL_ROUTE_DRIVE_FROM_TRANSIT | If the multimodal routing fails for a drive-from-transit trip; [description truncated in source] |
| 1011 | FAIL_ROUTE_TNC_AND_TRANSIT | |
| 1012 | FAIL_ROUTE_TNC | If the multimodal routing fails for a tnc trip; replan mode is triggered |
| 1013 | FAIL_ROUTE_SOV | Not used |
| 1014 | FAIL_ROUTE_MICROMOBILITY | |
| 1015 | NO_MOVE | If the origin and destination locations are the same, no movement needs to happen |
| 9999 | UNSIMULATED | If both the origin and destination locations fall into the no-simulation portion of [description truncated in source] |

## Enum: type (Planned_Activity.type)

| Value | Name | Description |
|---|---|---|
| -1 | DELETED | |
| -2 | EMPTY | |
| 0 | WORK_AT_HOME_ACTIVITY | |
| 1 | AT_HOME_ACTIVITY | |
| 2 | PRIMARY_WORK_ACTIVITY | |
| 3 | OTHER_WORK_ACTIVITY | |
| 4 | SCHOOL_ACTIVITY | |
| 5 | MAJOR_SHOPPING_ACTIVITY | |
| 6 | OTHER_SHOPPING_ACTIVITY | |
| 7 | SERVICE_VEHICLE_ACTIVITY | |
| 8 | HEALTHCARE_ACTIVITY | |
| 9 | PERSONAL_BUSINESS_ACTIVITY | |
| 10 | ERRANDS_ACTIVITY | |
| 11 | PICK_UP_OR_DROP_OFF_ACTIVITY | |
| 12 | EAT_OUT_ACTIVITY | |
| 13 | RELIGIOUS_OR_CIVIC_ACTIVITY | |
| 14 | LEISURE_ACTIVITY | |
| 15 | RECREATION_ACTIVITY | |
| 16 | SOCIAL_ACTIVITY | |
| 17 | CHANGE_TRANSPORTATION_ACTIVITY | |
| 18 | OTHER_ACTIVITY | |
| 22 | PART_TIME_WORK_ACTIVITY | |
| 23 | EV_CHARGING_ACTIVITY | |

---

# Table: Mode

> Dictionary of all modes supported for simulation within POLARIS. This table is currently only used for providing a text based representation of the hard-coded modes defined in POLARIS (Vehicle_Type_Keys enum).

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| mode_id * | INTEGER | NO | | | Unique identifier for mode, values at [mode_id] enum |
| mode_description | TEXT | NO | | | Text input describing the mode |

(\* = Primary key)

```sql
CREATE TABLE "Mode" (
  "mode_id" INTEGER NOT NULL PRIMARY KEY,
  "mode_description" TEXT NOT NULL)
```

## Enum: mode_id (Mode.mode_id)

| Value | Name | Description |
|---|---|---|
| 0 | SOV | Car driver |
| 1 | HOV_HH | Car passenger in household vehicle |
| 2 | HOV | Car passenger in household vehicle |
| 3 | TRUCK | |
| 4 | BUS | Mode identifier associated with an articulated 60 ft bus vehicle, if the [description truncated in source] |
| 5 | RAIL | Public transit with walk access and egress to any combination of rail sub-modes (subway, light [description truncated in source] |
| 7 | BICYCLE | Privately owned bicycle |
| 8 | WALK | Walking |
| 9 | TAXI | Taxi and TNC ride hailing for passengers |
| 10 | SCHOOLBUS | School bus |
| 11 | PARK_AND_RIDE | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 12 | KISS_AND_RIDE | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 13 | PARK_AND_RAIL | Public transit, accessed by driving and parking the car. The egress leg is walk. Can [description truncated in source] |
| 14 | KISS_AND_RAIL | Public transit, accessed by car with someone dropping off. The egress leg is walk. |
| 15 | TNC_AND_RIDE | Public transit, with access and/or egress by taxi or rideshare. |
| 17 | MD_TRUCK | Medium Duty Trucks, Classes 3-6 (10,001-26,000 lbs) according to FHWA classification |
| 18 | HD_TRUCK | Heavy Duty Trucks, Class 7-8 (> 26,001 lbs) according to FHWA classification |
| 19 | BPLATE | Pickup trucks and vans (Chicago model only), Class 2 (< 10,000 lbs) according to FHWA [description truncated in source] |
| 20 | LD_TRUCK | Light Duty Trucks, Class 2 (< 10,000 lbs) according to FHWA classification |
| 22 | BUS40 | |
| 23 | BUS60 | |
| 25 | RIDE_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can use any any [description truncated in source] |
| 26 | RIDE_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can use [description truncated in source] |
| 27 | RAIL_AND_UNPARK | Public transit, access is walk and egress is by parked car. Can only use rail modes [description truncated in source] |
| 28 | RAIL_AND_REKISS | Public transit, access is walk and egress is by being picked up by a car. Can only [description truncated in source] |
| 29 | MICROM | Docked micromobility mode (if micromobility and transit has bike time but no transit time) |
| 30 | MICROM_NODOCK | Dockless micromobility modes (if dockless micromobility and transit has bike time but [description truncated in source] |
| 31 | MICROM_AND_TRANSIT | Docked micromobility modes for access/egress to transit |
| 32 | MICROM_NODOCK_AND_TRANSIT | Dockless micromobility and transit modes combination |
| 33 | ODDELIVERY | Groceries and meals delivery |
| 999 | FAIL_MODE | If a traveler left home using drive-to-transit but returning home via transit and [description truncated in source] |
| 1000 | FAIL_ROUTE | Not used |
| 1001 | FAIL_REROUTE | If too many rerouting happens or the rerouting fails for a multimodal trip, the mode [description truncated in source] |
| 1002 | FAIL_UNPARK | If the traveler arrives to their parked vehicle on a return (transit-and-drive) trip [description truncated in source] |
| 1003 | FAIL_UNPARK2 | |
| 1004 | FAIL_MODE1 | |
| 1005 | FAIL_MODE2 | |
| 1006 | FAIL_MODE3 | |
| 1007 | FAIL_ROUTE_ACTIVE | If the multimodal routing fails for a walking or biking trip the mode becomes [description truncated in source] |
| 1008 | FAIL_ROUTE_WALK_AND_TRANSIT | |
| 1009 | FAIL_ROUTE_DRIVE_TO_TRANSIT | |
| 1010 | FAIL_ROUTE_DRIVE_FROM_TRANSIT | If the multimodal routing fails for a drive-from-transit trip; [description truncated in source] |
| 1011 | FAIL_ROUTE_TNC_AND_TRANSIT | |
| 1012 | FAIL_ROUTE_TNC | If the multimodal routing fails for a tnc trip; replan mode is triggered |
| 1013 | FAIL_ROUTE_SOV | Not used |
| 1014 | FAIL_ROUTE_MICROMOBILITY | |
| 1015 | NO_MOVE | If the origin and destination locations are the same, no movement needs to happen |
| 9999 | UNSIMULATED | If both the origin and destination locations fall into the no-simulation portion of [description truncated in source] |

---

# Table: Vehicle

> Vehicle table holds records for all vehicles simulated in POLARIS. This includes household-owned vehicles as well as fleet-owned vehicles.

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| vehicle_id * | INTEGER | NO | | | Unique identifier of this vehicle |
| hhold | INTEGER | NO | 0 | | Household to which this vehicle belongs. Negative values represent that the vehicle is fleet-owned. (foreign key to the Household table) |
| parking | INTEGER | NO | 0 | | Not used |
| L3_wtp | INTEGER | NO | 0 | | Not used |
| L4_wtp | INTEGER | NO | 0 | | Not used |
| type | INTEGER | NO | | Vehicle_Type(type) | Vehicle characteristics (foreign key to the Vehicle_Type table) |
| fleet | INTEGER | YES | | Fleet(fleet) | Fleet the vehicle belongs to, if any |
| subtype | INTEGER | NO | 0 | | Not used |

(\* = Primary key)

No enum tables on this page.

```sql
CREATE TABLE "Vehicle" (
  "vehicle_id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hhold" INTEGER NOT NULL DEFAULT 0,
  "parking" INTEGER NOT NULL DEFAULT 0,
  "L3_wtp" INTEGER NOT NULL DEFAULT 0,
  "L4_wtp" INTEGER NOT NULL DEFAULT 0,
  "type" INTEGER NOT NULL,
  "fleet" INTEGER NULL,
  "subtype" INTEGER NOT NULL DEFAULT 0,
  CONSTRAINT "type_fk"
    FOREIGN KEY ("type")
    REFERENCES "Vehicle_Type" ("type_id")
    DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT "fleet_fk"
    FOREIGN KEY ("fleet")
    REFERENCES "Fleet" ("fleet")
    DEFERRABLE INITIALLY DEFERRED);
```

---

# Table: Fleet

> Table denotes the different fleets operating within the simulation. Can be a TNC fleet, transit fleet, or freight-related fleet.

| Field | Type | NULL allowed | Default | Foreign key | Description |
|---|---|---|---|---|---|
| fleet * | INTEGER | NO | | | unique identifier denoting the fleet |
| name | TEXT | NO | '' | | text identifying the name of the fleet. Example includes "Operator_1" |

(\* = Primary key)

No enum tables on this page.

```sql
CREATE TABLE "Fleet" (
  "fleet" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "name" TEXT NOT NULL DEFAULT ''
)
```

---

## Notes / caveats

- The `Traveler` page states "THERE IS CURRENTLY NO DOCUMENTATION FOR THIS TABLE" — fields listed above but no descriptions.
- Several enum descriptions in the source docs are truncated mid-sentence (e.g. BUS, RAIL, PARK_AND_RIDE, FAIL_* entries); they are reproduced here exactly as published, flagged with `[description truncated in source]`. This affects only the long-mode enum which appears identically on Activity, Planned_Activity, Trip, and Mode pages.
- `Trip.vehicle` field-level FK is documented as `Vehicle(vehicle)` but the CREATE TABLE constraint is `REFERENCES "Vehicle" ("vehicle_id")`; the SQL is authoritative.
- The requested `https://polaris.taps.anl.gov/polaris-studio/demand/datamodel.html` returns 404; the demand data model overview page lives at `https://polaris.taps.anl.gov/polaris-studio/demand/auto_generated/datamodel.html`. It contains only conventions + a table of contents (no schema diagram).
- The demand database also contains tables not in the requested set (Person, Plan, Person_Gaps, TNC_* tables, Ev_Charging, Migrations, etc.) — see the Demand data model page for the full table list.
