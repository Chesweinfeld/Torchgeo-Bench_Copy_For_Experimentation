"""DEPRECATED — replaced by scripts/prepare_building_value.py.

Chaffee County's public parcel layer carries no assessed value fields, so this
county-specific script was dropped. The generalized geometry<->value-CSV join
(default county: Larimer, CO) lives in prepare_building_value.py.
"""
import sys

if __name__ == "__main__":
    sys.exit(
        "Deprecated. Use:\n"
        "  python scripts/prepare_building_value.py "
        "--parcels <geometry> --values <assessor.csv> --dry-run"
    )
