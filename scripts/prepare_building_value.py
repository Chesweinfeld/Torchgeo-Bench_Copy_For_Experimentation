#!/usr/bin/env python3
"""Build a *real* building-value regression dataset via a geometry <-> value join.

Why a join (not a single download)
-----------------------------------
County assessors almost always publish two separate things: parcel **geometry**
(a shapefile/GeoJSON/feature service) and assessed **values** (a CSV/table of
land value + improvement value), linked by a parcel/account ID. The public GIS
geometry layer usually does NOT carry dollar values. So this script joins them.

The regression target is *structure-only* value == the **improvement** value,
with land value dropped by construction (exactly the benchmark's design).

Default county: Larimer County, CO
----------------------------------
Larimer's Assessor Public Data Center publishes both pieces, joinable by account
number. Towns include Fort Collins, Loveland, Estes Park, Berthoud, Wellington;
the default bbox covers the smaller ones (Estes Park / Berthoud / Wellington) so
a run stays quick. Any county works — just supply its two files and, if needed,
add its column names to the candidate lists in CONFIG.

    Larimer Public Data Center: https://www.larimer.gov/assessor/publicdata
      - parcels (geometry): the parcel shapefile / GDB
      - values (table):     the Improvement + Land value export (has ACCOUNT + values)

Three layers this script joins
------------------------------
1. Parcel geometry  (--parcels)  : polygons/points with a parcel/account ID.
2. Assessor values  (--values)   : CSV/table with land + improvement value + ID.
3. NAIP imagery     (auto)       : R,G,B,N patches via Microsoft Planetary Computer.

Output (read by BuildingValueReal in datasets/building_value.py)
----------------------------------------------------------------
    data/building_value/<county>/
        {train,val,test}/patches.npy   float32 (N,C,H,W) DN ~0-3000
        {train,val,test}/labels.npy    float32 (N,2) -> [structure_value_usd, is_informal]
        manifest.json

Run
---
    pip install geopandas rasterio pystac-client planetary-computer pyproj pandas overturemaps
    # download the two Larimer files first (see URL above), then:
    python scripts/prepare_building_value.py \
        --parcels data/raw/larimer_parcels.shp \
        --values  data/raw/larimer_values.csv \
        --out     data/building_value/larimer_co --dry-run

--dry-run stops after the join and prints the resolved columns + split counts,
so you can confirm the schema before the imagery download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# CONFIG — county/source-specific knobs. Field lists are matched
# case-insensitively; the first column present wins. Add yours if missing.
# --------------------------------------------------------------------------- #
CONFIG = dict(
    county="larimer_co",
    # Smaller-town bbox in Larimer County, CO (WGS84 minx,miny,maxx,maxy):
    # Berthoud / Loveland fringe / Estes Park corridor. Widen for more data.
    aoi_bbox=(-105.65, 40.24, -105.05, 40.52),
    crs_metric="EPSG:32613",          # UTM 13N (meters), for patch windows
    patch_meters=40.0,
    patch_px=32,                      # must match BuildingValue.patch_size
    bands=("red", "green", "blue", "nir"),   # NAIP band order is R,G,B,N

    # --- join key (present in BOTH the geometry and the value table) ---------
    # Schedule number first (Larimer: geometry SCHEDNUM <-> value SCHEDULENUM);
    # both sides resolve independently but to the same identifier values.
    join_key=["SCHEDNUM", "SCHEDULENUM", "SCHEDULE", "PARCELNUM", "PARCELNO",
              "ACCOUNTNO", "ACCOUNT", "PARCEL", "PIN", "STRAP",
              "PARCEL_ID", "PARCELID"],

    # --- LONG-format value tables (one row per land/improvement component) ----
    # Larimer's value-detail is long: a VALUETYPE column marks each row as Land
    # or Improvement and ACTUALVALUE holds the dollars. If both columns below are
    # present we aggregate improvement rows per account. Otherwise we fall back
    # to a wide single-column improvement value (field_improvement_value).
    value_type_col=["VALUETYPE", "ABSTRACTTYPE"],
    value_amount_col=["ACTUALVALUE", "ACTUAL_VALUE", "MARKETVALUE", "TOTALVALUE"],
    improvement_markers=["improvement", "i", "imp", "improvements"],  # case-insensitive
    land_markers=["land", "l"],

    # --- WIDE-format fallback (single improvement value column) --------------
    field_improvement_value=["IMPROVEMENT_VALUE", "IMPROVEMENTS", "IMP_VALUE",
                             "IMPROVEMEN", "IMPROVVAL", "IMP_VAL", "STRUCT_VAL",
                             "BLDG_VALUE", "BUILDINGVALUE"],
    field_land_value=["LAND_VALUE", "LANDVALUE", "LAND_VAL", "LANDVAL"],
    field_land_use=["LAND_USE", "LANDUSE", "PROPERTY_TYPE", "PROPCLASS", "LU_DESC"],

    # --- footprints ----------------------------------------------------------
    footprints_source="parcels",      # "overture" | "ms" | "parcels"
    # "parcels" = center each patch on the parcel's representative point (no
    # extra download; good first run). "overture"/"ms" = center on real building
    # footprints joined into parcels (more accurate; needs the extra dep).

    # --- splits --------------------------------------------------------------
    split_grid_deg=0.03,              # ~3 km blocks -> whole blocks per split
    split_fracs=dict(train=0.7, val=0.15, test=0.15),
    max_buildings=8000,               # cap so a first run stays quick
    random_seed=17,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pick(columns, candidates):
    """First candidate present in columns, matched case-insensitively."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _read_any(path):
    """Read a geo file OR a flat table (CSV) into a (Geo)DataFrame."""
    import pandas as pd

    p = str(path).lower()
    if p.endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if p.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep, dtype=str)   # keep IDs as strings
    import geopandas as gpd
    return gpd.read_file(path)


def _structure_value_per_account(cfg, vals, vk):
    """Collapse a value table down to one structure value per join key.

    Handles both layouts:
      * LONG  (Larimer): rows tagged Land/Improvement via value_type_col, dollars
        in value_amount_col — sum the improvement rows per account.
      * WIDE  : a single improvement-value column (field_improvement_value).
    Returns a DataFrame with columns ['_key', 'structure_value_usd'].
    """
    import pandas as pd

    vals = vals.copy()
    vals["_key"] = vals[vk].astype(str).str.strip()

    tcol = _pick(vals.columns, cfg["value_type_col"])
    acol = _pick(vals.columns, cfg["value_amount_col"])
    if tcol and acol:
        markers = {m.lower() for m in cfg["improvement_markers"]}
        is_imp = vals[tcol].astype(str).str.strip().str.lower().isin(markers)
        imp = vals[is_imp].copy()
        imp["_amt"] = pd.to_numeric(imp[acol], errors="coerce").fillna(0.0)
        out = (imp.groupby("_key", as_index=False)["_amt"].sum()
               .rename(columns={"_amt": "structure_value_usd"}))
        print(f"    long-format value table: type={tcol!r} amount={acol!r} "
              f"(kept {int(is_imp.sum()):,} improvement rows)")
        return out

    wide = _pick(vals.columns, cfg["field_improvement_value"])
    if wide is None:
        raise SystemExit(
            "No improvement value found. Need either a long-format pair "
            "(value_type_col + value_amount_col) or a wide improvement column.\n"
            f"  value columns: {list(vals.columns)}"
        )
    vals["structure_value_usd"] = pd.to_numeric(vals[wide], errors="coerce")
    print(f"    wide-format value column: {wide!r}")
    return vals[["_key", "structure_value_usd"]]


def _load_joined_parcels(cfg, parcels_path, values_path):
    """Join parcel geometry to the assessor value table on a shared ID."""
    import geopandas as gpd

    geom = _read_any(parcels_path)
    if not isinstance(geom, gpd.GeoDataFrame):
        raise SystemExit("--parcels must be a geometry file (shp/geojson/gdb).")
    vals = _read_any(values_path)

    gk = _pick(geom.columns, cfg["join_key"])
    vk = _pick(vals.columns, cfg["join_key"])
    if gk is None or vk is None:
        raise SystemExit(
            "Could not find a shared join key.\n"
            f"  geometry columns: {list(geom.columns)}\n"
            f"  value columns:    {list(vals.columns)}\n"
            "Add the matching ID name to join_key in CONFIG."
        )

    val_by_key = _structure_value_per_account(cfg, vals, vk)
    geom["_key"] = geom[gk].astype(str).str.strip()
    merged = geom.merge(val_by_key, on="_key", how="inner")
    merged = merged[merged["structure_value_usd"] > 0].copy()

    print(f"    join key: geometry {gk!r} <-> value {vk!r}")
    print(f"    {len(merged):,} parcels joined with positive structure value")
    if merged.empty:
        raise SystemExit("Join produced 0 rows — the IDs likely need normalizing "
                         "(leading zeros, dashes). Inspect both _key columns.")
    return merged.to_crs("EPSG:4326")


def _building_points(cfg, parcels):
    import geopandas as gpd

    if cfg["footprints_source"] == "parcels":
        pts = parcels.copy()
        pts["geometry"] = pts.geometry.representative_point()
    else:
        fp = _load_footprints(cfg)
        fp["geometry"] = fp.geometry.representative_point()
        pts = gpd.sjoin(fp, parcels, predicate="within", how="inner")

    pts = pts.dropna(subset=["structure_value_usd"])
    # US parcels have no informal-settlement registry -> flag 0 (formal). The
    # informal/formal robustness slice activates only with informal-city data.
    pts["is_informal"] = 0.0
    # keep only points inside the AOI bbox
    minx, miny, maxx, maxy = cfg["aoi_bbox"]
    inx = pts.geometry.x.between(minx, maxx) & pts.geometry.y.between(miny, maxy)
    pts = pts[inx]
    if len(pts) > cfg["max_buildings"]:
        pts = pts.sample(cfg["max_buildings"], random_state=cfg["random_seed"])
    return pts.reset_index(drop=True)


def _load_footprints(cfg):
    if cfg["footprints_source"] == "overture":
        from overturemaps import core as ov  # type: ignore
        return ov.geodataframe("building", bbox=cfg["aoi_bbox"])
    raise NotImplementedError(
        "MS footprints: reuse the mercantile/quadkey loader from "
        "example_building_footprints.ipynb, then clip to aoi_bbox."
    )


def _assign_splits(cfg, lon, lat):
    rng = np.random.default_rng(cfg["random_seed"])
    g = cfg["split_grid_deg"]
    cell = np.floor(lon / g).astype(int) * 100003 + np.floor(lat / g).astype(int)
    uniq = np.unique(cell); rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(cfg["split_fracs"]["train"] * n)
    n_va = int(cfg["split_fracs"]["val"] * n)
    role = {c: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
            for i, c in enumerate(uniq)}
    return np.array([role[c] for c in cell])


def _read_patches(cfg, pts, cache_dir):
    """Read NAIP patches, checkpointing to disk so an interrupted run resumes.

    Patches accumulate in a memmap (``patches.dat``) with a boolean ``filled``
    mask flushed periodically. Re-running skips rows already fetched, so a
    closed laptop / dropped connection only costs the in-flight batch.
    """
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    n, c, px = len(pts), len(cfg["bands"]), cfg["patch_px"]
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    mmap_path, mask_path = cache_dir / "patches.dat", cache_dir / "filled.npy"
    shape = (n, c, px, px)
    out = np.memmap(mmap_path, dtype="float32", mode=("r+" if mmap_path.exists() else "w+"),
                    shape=shape)
    filled = (np.load(mask_path) if mask_path.exists()
              else np.zeros(n, dtype=bool))
    if filled.shape[0] != n:                       # cache from a different run
        filled = np.zeros(n, dtype=bool); out[:] = 0
    if filled.any():
        print(f"    resuming: {int(filled.sum()):,}/{n:,} patches already cached")

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = list(cat.search(collections=["naip"], bbox=cfg["aoi_bbox"]).items())
    if not items:
        raise RuntimeError("No NAIP items for AOI — check bbox/collection.")
    to_m = Transformer.from_crs("EPSG:4326", cfg["crs_metric"], always_xy=True)
    half = cfg["patch_meters"] / 2.0
    xs, ys = to_m.transform(pts.geometry.x.values, pts.geometry.y.values)

    since_flush = 0
    for item in items:
        if filled.all():
            break
        with rasterio.open(item.assets["image"].href) as src:
            for i, (x, y) in enumerate(zip(xs, ys)):
                if filled[i]:
                    continue
                win = from_bounds(x - half, y - half, x + half, y + half,
                                  transform=src.transform)
                try:
                    arr = src.read(window=win, out_shape=(c, px, px),
                                   boundless=True, fill_value=0)
                except Exception:
                    continue
                if arr.shape[0] >= c and arr.any():
                    out[i] = arr[:c].astype(np.float32)
                    filled[i] = True
                    since_flush += 1
                    if since_flush >= 500:         # checkpoint periodically
                        out.flush(); np.save(mask_path, filled); since_flush = 0
        out.flush(); np.save(mask_path, filled)    # checkpoint per NAIP tile
    print(f"    fetched {int(filled.sum()):,}/{n:,} patches "
          f"({n - int(filled.sum()):,} outside NAIP coverage -> left zero)")
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcels", required=True, help="Parcel geometry file (shp/geojson/gdb).")
    ap.add_argument("--values", required=True, help="Assessor value table (csv) with land+improvement value.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Join + splits only; skip imagery download.")
    args = ap.parse_args()
    cfg = CONFIG
    out = Path(args.out or f"data/building_value/{cfg['county']}")

    print("1/4 joining parcels <-> assessor values …")
    parcels = _load_joined_parcels(cfg, args.parcels, args.values)

    print("2/4 building points …")
    pts = _building_points(cfg, parcels)
    print(f"    {len(pts):,} building patches to cut (within AOI, capped)")

    splits = _assign_splits(cfg, pts.geometry.x.values, pts.geometry.y.values)
    if args.dry_run:
        for s in ("train", "val", "test"):
            print(f"    {s}: {(splits == s).sum():,}")
        print("dry-run: stopping before imagery download.")
        return

    print("3/4 reading NAIP patches (R,G,B,N) …")
    patches = _read_patches(cfg, pts, out / "_cache")
    labels = np.stack([pts["structure_value_usd"].to_numpy(np.float32),
                       pts["is_informal"].to_numpy(np.float32)], axis=1)

    print("4/4 writing splits …")
    for s in ("train", "val", "test"):
        m = splits == s
        d = out / s; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "patches.npy", patches[m])
        np.save(d / "labels.npy", labels[m])
    (out / "manifest.json").write_text(json.dumps({
        "county": cfg["county"],
        "source_imagery": "NAIP via Microsoft Planetary Computer",
        "source_value": "county assessor improvement value (land removed)",
        "source_footprints": cfg["footprints_source"],
        "target": "structure_value_usd",
        "bands": list(cfg["bands"]),
        "patch_px": cfg["patch_px"],
        "counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "note": "is_informal is 0 for all US parcels; slice activates with informal-city data.",
    }, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
