"""Create small per-country sample files in every supported format.

They are used to demo the upload and the retraining flow: load one country,
train, then append another one and retrain.

Usage:
    python -m scripts.make_samples
"""

from __future__ import annotations

from michelin.config import COL_LOCATION, DEFAULT_DATASET, SAMPLES_DIR
from michelin.data.loader import read_path, write_table

# (country as it appears at the end of "Location", output file name)
SAMPLES = (
    ("France", "france.csv"),
    ("Japan", "japan.json"),
    ("USA", "usa.xml"),
    ("Italy", "italy.csv"),
    ("Spain", "spain.json"),
    ("Brazil", "brazil.xml"),
)


def main() -> None:
    df = read_path(DEFAULT_DATASET)
    country = df[COL_LOCATION].str.split(",").str[-1].str.strip()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_name in SAMPLES:
        subset = df[country == name]
        write_table(subset, SAMPLES_DIR / file_name)
        print(f"{file_name}: {len(subset):,} rows")


if __name__ == "__main__":
    main()
