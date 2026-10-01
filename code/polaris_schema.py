"""POLARIS-compatible demand schema for the SEQ framework.

Single source of truth for the Polaris-style SQLite demand store:
  Household -> Person -> Plan -> Trip -> Activity
plus the Mode / Activity_Type dictionary tables and the enums they document.

Source: Polaris 26.03 docs (polaris.taps.anl.gov/polaris-studio/demand/auto_generated/),
field-for-field. Times are seconds of simulation day; distances metres.
Extended columns from the SEQ pipeline (census codes, latent vars, accessibility,
GA features, ...) are preserved verbatim as extra columns on the same rows, so
the converter never loses information.
"""

# ── Polaris enums ────────────────────────────────────────────────────────────

# Mode codes (Vehicle_Type_Keys). Full list from the Mode / Activity /
# Trip / Planned_Activity docs.
POLARIS_MODE = {
    0: 'SOV', 1: 'HOV_HH', 2: 'HOV', 3: 'TRUCK',
    4: 'BUS', 5: 'RAIL',
    7: 'BICYCLE', 8: 'WALK', 9: 'TAXI', 10: 'SCHOOLBUS',
    11: 'PARK_AND_RIDE', 12: 'KISS_AND_RIDE', 13: 'PARK_AND_RAIL',
    14: 'KISS_AND_RAIL', 15: 'TNC_AND_RIDE',
    17: 'MD_TRUCK', 18: 'HD_TRUCK', 19: 'BPLATE', 20: 'LD_TRUCK',
    22: 'BUS40', 23: 'BUS60',
    25: 'RIDE_AND_UNPARK', 26: 'RIDE_AND_REKISS',
    27: 'RAIL_AND_UNPARK', 28: 'RAIL_AND_REKISS',
    29: 'MICROM', 30: 'MICROM_NODOCK', 31: 'MICROM_AND_TRANSIT',
    32: 'MICROM_NODOCK_AND_TRANSIT', 33: 'ODDELIVERY',
    999: 'FAIL_MODE', 1000: 'FAIL_ROUTE', 1001: 'FAIL_REROUTE',
    1002: 'FAIL_UNPARK', 1003: 'FAIL_UNPARK2', 1004: 'FAIL_MODE1',
    1005: 'FAIL_MODE2', 1006: 'FAIL_MODE3', 1007: 'FAIL_ROUTE_ACTIVE',
    1008: 'FAIL_ROUTE_WALK_AND_TRANSIT', 1009: 'FAIL_ROUTE_DRIVE_TO_TRANSIT',
    1010: 'FAIL_ROUTE_DRIVE_FROM_TRANSIT', 1011: 'FAIL_ROUTE_TNC_AND_TRANSIT',
    1012: 'FAIL_ROUTE_TNC', 1013: 'FAIL_ROUTE_SOV',
    1014: 'FAIL_ROUTE_MICROMOBILITY', 1015: 'NO_MOVE', 9999: 'UNSIMULATED',
}
MODE_NAME_TO_CODE = {name: code for code, name in POLARIS_MODE.items()}

# Activity types (Activity.type / Planned_Activity.type).
POLARIS_ACTIVITY_TYPE = {
    -1: 'DELETED', -2: 'EMPTY',
    0: 'WORK_AT_HOME_ACTIVITY', 1: 'AT_HOME_ACTIVITY',
    2: 'PRIMARY_WORK_ACTIVITY', 3: 'OTHER_WORK_ACTIVITY',
    4: 'SCHOOL_ACTIVITY', 5: 'MAJOR_SHOPPING_ACTIVITY',
    6: 'OTHER_SHOPPING_ACTIVITY', 7: 'SERVICE_VEHICLE_ACTIVITY',
    8: 'HEALTHCARE_ACTIVITY', 9: 'PERSONAL_BUSINESS_ACTIVITY',
    10: 'ERRANDS_ACTIVITY', 11: 'PICK_UP_OR_DROP_OFF_ACTIVITY',
    12: 'EAT_OUT_ACTIVITY', 13: 'RELIGIOUS_OR_CIVIC_ACTIVITY',
    14: 'LEISURE_ACTIVITY', 15: 'RECREATION_ACTIVITY',
    16: 'SOCIAL_ACTIVITY', 17: 'CHANGE_TRANSPORTATION_ACTIVITY',
    18: 'OTHER_ACTIVITY', 22: 'PART_TIME_WORK_ACTIVITY',
    23: 'EV_CHARGING_ACTIVITY',
}
ACTIVITY_NAME_TO_CODE = {name: code for code, name in POLARIS_ACTIVITY_TYPE.items()}

# Household types (Household.type).
POLARIS_HOUSEHOLD_TYPE = {
    1: 'HHT_OWN_FAMILY_MARRIED', 2: 'HHT_OWN_FAMILY_MALE',
    3: 'HHT_OWN_FAMILY_FEMALE', 4: 'HHT_OWN_NONFAMILY_SINGLE',
    5: 'HHT_OWN_NONFAMILY_OTHER', 6: 'HHT_RENT_FAMILY_MARRIED',
    7: 'HHT_RENT_FAMILY_MALE', 8: 'HHT_RENT_FAMILY_FEMALE',
    9: 'HHT_RENT_NONFAMILY_SINGLE', 10: 'HHT_RENT_NONFAMILY_OTHER',
    11: 'HHT_GQ',
}

# Housing unit types (Household.housing_unit_type).
POLARIS_HOUSING_UNIT_TYPE = {
    1: 'HU_SINGLE_FAMILY', 2: 'HU_SMALL_APT_BLD', 3: 'HU_LARGE_APT_BLD',
    4: 'HU_OTHER', 5: 'HU_GROUP_QUARTERS',
}

# Trip types (Trip.type).
POLARIS_TRIP_TYPE = {
    -1: 'NULLTRIP', 11: 'ABM', 22: 'EXTERNAL', 32: 'TNC_VEHICLE',
    33: 'TNC_REQUEST', 44: 'FREIGHT', 45: 'FREIGHT_AV', 55: 'TRANSIT',
    99: 'UNSIMULATED',
}

# Worker class (Person.worker_class).
POLARIS_WORKER_CLASS = {
    0: 'COW_NA', 1: 'COW_EMPLOYEE', 2: 'COW_EMPLOYEE_NONPROFIT',
    3: 'COW_EMPLOYEE_LOCAL_GOVERNMENT', 4: 'COW_EMPLOYEE_STATE_GOVERNMENT',
    5: 'COW_EMPLOYEE_FEDERAL_GOVERNMENT', 6: 'COW_SELF_EMPLOYED_UNINCORPORATED',
    7: 'COW_SELF_EMPLOYED_INCORPORATED', 8: 'COW_WORK_FOR_FAMILY',
    9: 'COW_UNEMPLOYED',
}

# Employment status (Person.employment, ACS ESR).
POLARIS_EMPLOYMENT = {
    0: 'EMPLOYMENT_STATUS_NA', 1: 'EMPLOYMENT_STATUS_CIVILIAN_AT_WORK',
    2: 'EMPLOYMENT_STATUS_CIVILIAN_NOT_AT_WORK',
    3: 'EMPLOYMENT_STATUS_UNEMPLOYED',
    4: 'EMPLOYMENT_STATUS_ARMED_FORCES_AT_WORK',
    5: 'EMPLOYMENT_STATUS_ARMED_FORCES_NOT_AT_WORK',
    6: 'EMPLOYMENT_STATUS_NOT_IN_LABOR_FORCE',
}

# Gender (Person.gender).
POLARIS_GENDER = {0: 'NA', 1: 'MALE', 2: 'FEMALE'}

# Marital status (Person.marital_status).
POLARIS_MARITAL_STATUS = {
    1: 'MARRIED', 2: 'WIDOWED', 3: 'DIVORCED', 4: 'SEPARATED',
    5: 'NEVER_MARRIED',
}

# ── SEQ framework -> Polaris mappings ────────────────────────────────────────

# stage6 mode_name -> Polaris mode code.
SEQ_MODE_MAP = {
    'DA': 0,            # SOV
    'Other': 2,         # HOV / shared ride (Other-SR grouping)
    'Transit': 4,       # BUS (SEQ transit incl. rail; override via config)
    'Bike': 7,          # BICYCLE
    'Walk': 8,          # WALK
}

# stage6 purpose -> Polaris activity type code.
SEQ_PURPOSE_MAP = {
    'work': 2,             # PRIMARY_WORK_ACTIVITY
    'edu': 4,              # SCHOOL_ACTIVITY
    'shopping': 5,         # MAJOR_SHOPPING_ACTIVITY
    'othmaint': 6,         # OTHER_SHOPPING_ACTIVITY
    'healthcare': 8,       # HEALTHCARE_ACTIVITY
    'personal_business': 9,# PERSONAL_BUSINESS_ACTIVITY
    'errands': 10,         # ERRANDS_ACTIVITY
    'escort': 11,          # PICK_UP_OR_DROP_OFF_ACTIVITY
    'eat': 12,             # EAT_OUT_ACTIVITY
    'leisure': 14,         # LEISURE_ACTIVITY
    'recreation': 15,      # RECREATION_ACTIVITY
    'social': 16,          # SOCIAL_ACTIVITY
    'other': 18,           # OTHER_ACTIVITY
    'home': 1,             # AT_HOME_ACTIVITY
}

# ── Canonical table definitions ──────────────────────────────────────────────
# name -> (pk, ordered list of (column, type, notnull, default, fk))
# Mirrors the Polaris CREATE TABLE statements. Converter fills the canonical
# columns; any remaining source columns are appended verbatim (extended cols).

HOUSEHOLD_TABLE = ('Household', [
    ('household', 'INTEGER', True, None, None),          # PK
    ('hhold', 'INTEGER', True, None, None),              # upstream (PUMS) id
    ('location', 'INTEGER', True, None, 'Location'),     # home zone -> ORDER
    ('persons', 'INTEGER', False, 0, None),
    ('workers', 'INTEGER', False, 0, None),
    ('vehicles', 'INTEGER', False, 0, None),
    ('type', 'INTEGER', False, 0, None),                 # POLARIS_HOUSEHOLD_TYPE
    ('income', 'INTEGER', False, 0, None),
    ('bikes', 'INTEGER', False, 0, None),
    ('housing_unit_type', 'INTEGER', False, 0, None),
    ('ecom', 'INTEGER', False, 0, None),
    ('delRat', 'REAL', True, 0, None),
    ('dispose_veh', 'INTEGER', False, 0, None),
    ('time_in_home', 'REAL', False, 0, None),
    ('Has_Residential_Charging', 'INTEGER', False, 0, None),
    ('num_groceries', 'INTEGER', False, 0, None),
    ('num_meals', 'INTEGER', False, 0, None),
])

PERSON_TABLE = ('Person', [
    ('person', 'INTEGER', True, None, None),             # PK
    ('id', 'INTEGER', False, 0, None),                   # index within household
    ('school_location_id', 'INTEGER', False, 0, None),
    ('work_location_id', 'INTEGER', False, 0, None),
    ('age', 'INTEGER', False, 0, None),
    ('worker_class', 'INTEGER', False, 0, None),
    ('education', 'INTEGER', False, 0, None),
    ('industry', 'INTEGER', False, 0, None),
    ('employment', 'INTEGER', False, 0, None),
    ('gender', 'INTEGER', False, 0, None),
    ('income', 'INTEGER', False, 0, None),
    ('journey_to_work_arrival_time', 'INTEGER', False, 0, None),
    ('journey_to_work_mode', 'INTEGER', False, 0, None),
    ('journey_to_work_travel_time', 'INTEGER', False, 0, None),
    ('journey_to_work_vehicle_occupancy', 'INTEGER', False, 0, None),
    ('marital_status', 'INTEGER', False, 0, None),
    ('race', 'INTEGER', False, 0, None),
    ('school_enrollment', 'INTEGER', False, 0, None),
    ('school_grade_level', 'INTEGER', False, 0, None),
    ('work_hours', 'INTEGER', False, 0, None),
    ('telecommute_level', 'INTEGER', False, 0, None),
    ('transit_pass', 'INTEGER', False, 0, None),
    ('household', 'INTEGER', True, None, 'Household'),   # FK
    ('time_in_job', 'REAL', False, 0, None),
    ('disability', 'INTEGER', False, 2, None),
    ('escooter_use_level', 'INTEGER', False, 0, None),
    ('is_long_term_chooser', 'INTEGER', False, 1, None),
])

PLAN_TABLE = ('Plan', [
    ('plan', 'INTEGER', True, None, None),               # PK
    ('depart', 'REAL', True, 0, None),
    ('arrive', 'REAL', True, 0, None),
    ('activity', 'REAL', True, 0, None),
    ('walk', 'REAL', True, 0, None),
    ('drive', 'REAL', True, 0, None),
    ('transit', 'REAL', True, 0, None),
    ('wait', 'REAL', True, 0, None),
    ('other', 'REAL', True, 0, None),
    ('length', 'REAL', True, 0, None),
    ('cost', 'REAL', True, 0, None),
    ('impedance', 'INTEGER', False, 0, None),
    ('leg_mode', 'INTEGER', False, 0, None),
    ('leg_type', 'INTEGER', False, 0, None),
    ('leg_id', 'INTEGER', False, 0, None),
    ('leg_time', 'REAL', True, 0, None),
    ('leg_length', 'REAL', True, 0, None),
    ('leg_cost', 'REAL', True, 0, None),
    ('leg_imp', 'INTEGER', False, 0, None),
])

TRIP_TABLE = ('Trip', [
    ('trip_id', 'INTEGER', True, None, None),            # PK
    ('hhold', 'INTEGER', False, 0, None),
    ('path', 'INTEGER', False, -1, None),
    ('path_multimodal', 'INTEGER', False, -1, None),
    ('tour', 'INTEGER', False, 0, None),
    ('trip', 'INTEGER', False, 0, None),
    ('start', 'REAL', True, 0, None),
    ('end', 'REAL', True, 0, None),
    ('duration', 'REAL', True, 0, None),
    ('experienced_gap', 'REAL', True, 0, None),
    ('origin', 'INTEGER', False, 0, None),               # Location
    ('destination', 'INTEGER', False, 0, None),          # Location
    ('purpose', 'INTEGER', False, 0, None),
    ('mode', 'INTEGER', False, 0, None),
    ('constraint', 'INTEGER', False, 0, None),
    ('priority', 'INTEGER', False, 0, None),
    ('vehicle', 'INTEGER', True, None, 'Vehicle'),
    ('passengers', 'INTEGER', False, 0, None),
    ('type', 'INTEGER', False, 0, None),                 # POLARIS_TRIP_TYPE
    ('partition', 'INTEGER', False, 0, None),
    ('person', 'INTEGER', True, None, 'Person'),
    ('travel_distance', 'REAL', True, 0, None),
    ('skim_travel_time', 'REAL', True, 0, None),
    ('routed_travel_time', 'REAL', True, 0, None),
    ('access_egress_ovtt', 'REAL', True, 0, None),
    ('toll', 'REAL', True, 0, None),
    ('has_artificial_trip', 'INTEGER', False, 0, None),
    ('number_of_switches', 'INTEGER', False, 0, None),
    ('request', 'INTEGER', False, 0, None),
    ('monetary_cost', 'REAL', True, 0, None),
    ('initial_energy_level', 'REAL', True, 0, None),
    ('final_energy_level', 'REAL', True, 0, None),
])

ACTIVITY_TABLE = ('Activity', [
    ('id', 'INTEGER', True, None, None),                 # PK
    ('seq_num', 'INTEGER', False, 0, None),
    ('location_id', 'INTEGER', False, 0, None),          # Location
    ('start_time', 'REAL', True, 0, None),
    ('duration', 'REAL', True, 0, None),
    ('mode', 'TEXT', False, '', None),                   # mode name
    ('type', 'TEXT', False, '', None),                   # activity type name
    ('person', 'INTEGER', True, None, 'Person'),         # FK
    ('trip', 'INTEGER', False, 0, None),                 # trip that reached it
    ('origin_id', 'INTEGER', False, 0, None),            # previous activity loc
])

MODE_DICT_TABLE = ('Mode', [
    ('mode_id', 'INTEGER', True, None, None),
    ('mode_description', 'TEXT', True, None, None),
])

ACTIVITY_TYPE_DICT_TABLE = ('Activity_Type', [
    ('activity_type_id', 'INTEGER', True, None, None),
    ('activity_type_description', 'TEXT', True, None, None),
])

LOCATION_TABLE = ('Location', [
    ('location', 'INTEGER', True, None, None),           # = zone ORDER
    ('SA2_MAIN16', 'TEXT', True, None, None),
    ('SA2_NAME16', 'TEXT', True, None, None),
    ('longitude', 'REAL', True, None, None),
    ('latitude', 'REAL', True, None, None),
])

ALL_TABLES = [
    HOUSEHOLD_TABLE, PERSON_TABLE, PLAN_TABLE, TRIP_TABLE,
    ACTIVITY_TABLE, MODE_DICT_TABLE, ACTIVITY_TYPE_DICT_TABLE,
    LOCATION_TABLE,
]

# Canonical columns per table (used to separate canonical vs extended).
CANONICAL_COLS = {name: [c[0] for c in cols] for name, cols in ALL_TABLES}


def create_ddl():
    """Return a list of CREATE TABLE statements for the canonical tables."""
    stmts = []
    for name, cols in ALL_TABLES:
        col_defs = []
        for col, typ, nn, default, fk in cols:
            d = f'"{col}" {typ}'
            if nn:
                d += ' NOT NULL'
            if default is not None:
                if isinstance(default, str):
                    d += f" DEFAULT '{default}'"
                else:
                    d += f' DEFAULT {default}'
            col_defs.append(d)
        stmts.append(f'CREATE TABLE IF NOT EXISTS "{name}" (\n  '
                     + ',\n  '.join(col_defs) + ')')
    return stmts
