"""Project-wide constants: paths, raw column names and derived column names.

Keeping every column name in one place means the rest of the code never
hard-codes strings, and a schema change only needs to be fixed here.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET = DATA_DIR / "michelin_restaurants.csv"

SUPPORTED_EXTENSIONS = (".csv", ".json", ".xml")

# --- Raw columns (as published in the Michelin Guide dataset) ----------------

COL_NAME = "Name"
COL_ADDRESS = "Address"
COL_LOCATION = "Location"
COL_PRICE = "Price"
COL_CUISINE = "Cuisine"
COL_LONGITUDE = "Longitude"
COL_LATITUDE = "Latitude"
COL_AWARD = "Award"
COL_GREEN_STAR = "GreenStar"
COL_FACILITIES = "FacilitiesAndServices"

# Columns an uploaded file must contain to be analysed. Optional columns
# (phone, URLs, description) are kept when present but never required.
REQUIRED_COLUMNS = (
    COL_NAME,
    COL_LOCATION,
    COL_PRICE,
    COL_CUISINE,
    COL_LONGITUDE,
    COL_LATITUDE,
    COL_AWARD,
    COL_GREEN_STAR,
    COL_FACILITIES,
)

# --- Derived columns (created by the cleaning step) --------------------------

COL_CITY = "City"
COL_COUNTRY = "Country"
COL_PRICE_LEVEL = "PriceLevel"
COL_FACILITY_LIST = "FacilityList"
COL_FACILITY_COUNT = "FacilityCount"
COL_IS_STARRED = "IsStarred"

# --- Domain values -------------------------------------------------------------

AWARD_ORDER = ("Selected Restaurants", "Bib Gourmand", "1 Star", "2 Stars", "3 Stars")
STAR_AWARDS = ("1 Star", "2 Stars", "3 Stars")

# Some locations are a single token (e.g. "Dubai"); map them to their country.
COUNTRY_ALIASES = {
    "Dubai": "United Arab Emirates",
    "Abu Dhabi": "United Arab Emirates",
}

# Fixed palette so every chart shows the same award with the same color.
AWARD_COLORS = {
    "Selected Restaurants": "#9AA5B1",
    "Bib Gourmand": "#E8A33D",
    "1 Star": "#C8102E",
    "2 Stars": "#8E0C20",
    "3 Stars": "#4A0612",
    "Starred": "#C8102E",
    "Not starred": "#9AA5B1",
}

RANDOM_STATE = 42
