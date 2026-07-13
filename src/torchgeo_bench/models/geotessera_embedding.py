"""GeoTessera precomputed-embedding lookup model for torchgeo-bench.

Unlike every other model in this package, :class:`GeoTesseraEmbeddingBenchModel`
does not run a forward pass on pixels. Tessera (ucam-eo/tessera) is a
foundation model trained on full-year Sentinel-1/Sentinel-2 time series;
its trained weights aren't publicly downloadable, but its published
per-pixel embeddings are (CC0, anonymous S3) via the ``geotessera`` package.
This model looks up the precomputed embedding at each sample's location and
year instead of re-deriving one from a single-timestamp chip.

Requires the dataset to declare a matching
:attr:`~torchgeo_bench.datasets.base.BenchDataset.geo_fields` (at least
``lat``/``lon``; ``year`` is optional — falls back to :attr:`default_year`).
"""

import logging

import torch

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)


def _build_client(store_url: str | None):
    try:
        from geotessera.store import GeoTesseraZarr
    except ImportError as e:
        raise ImportError(
            "geotessera is required for GeoTesseraEmbeddingBenchModel; install "
            "with `pip install torchgeo-bench[geotessera]`."
        ) from e
    return GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)


class GeoTesseraEmbeddingBenchModel(BenchModel):
    """Looks up precomputed Tessera embeddings by location/year.

    Args:
        bands: Ordered list of :class:`BandSpec`. Required by the
            :class:`BenchModel` constructor contract, but otherwise unused —
            this model ignores pixel content entirely.
        default_year: Year to use when a sample's ``geo["year"]`` is missing
            (``NaN``) or the dataset didn't supply a ``year`` field at all.
        store_url: GeoTessera zarr store URL. ``None`` uses the package
            default (the public TESSERA v1 store).
    """

    requires_geolocation = True

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        default_year: int = 2024,
        store_url: str | None = None,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization="identity")
        self.default_year = default_year
        self._client = _build_client(store_url)
        logger.info(
            "GeoTesseraEmbeddingBenchModel initialized (default_year=%d) — "
            "ignores pixel content and looks up precomputed embeddings by "
            "location instead.",
            default_year,
        )

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "GeoTesseraEmbeddingBenchModel requires geolocation; use a dataset "
            "with geo_fields set (at least 'lat'/'lon')."
        )

    def _forward_patch_features_geo(
        self, images: torch.Tensor, geo: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        if geo is None or "lat" not in geo or "lon" not in geo:
            raise ValueError(
                "GeoTesseraEmbeddingBenchModel requires the dataset to supply "
                "'lat'/'lon' via BenchDataset.geo_fields; got "
                f"{sorted(geo) if geo else None}."
            )
        batch_size = images.shape[0]
        lats = geo["lat"].tolist()
        lons = geo["lon"].tolist()
        years = geo["year"].tolist() if "year" in geo else [float("nan")] * batch_size
        resolved_years = [int(y) if y == y else self.default_year for y in years]

        # sample_points() takes one year per call — group same-year samples
        # to minimize round-trips to the zarr store.
        by_year: dict[int, list[int]] = {}
        for i, year in enumerate(resolved_years):
            by_year.setdefault(year, []).append(i)

        rows: list[torch.Tensor | None] = [None] * batch_size
        for year, indices in by_year.items():
            points = [(lons[i], lats[i]) for i in indices]
            embeddings = self._client.sample_points(points, year=year, progress=False)
            for row_idx, sample_idx in enumerate(indices):
                rows[sample_idx] = torch.as_tensor(embeddings[row_idx], dtype=torch.float32)

        out = torch.stack(rows)  # type: ignore[arg-type]

        if torch.isnan(out).any():
            n_nan = torch.isnan(out).any(dim=1).sum().item()
            logger.warning(
                "%d/%d samples had no GeoTessera coverage at their location/year "
                "and returned NaN embeddings.",
                n_nan,
                batch_size,
            )

        return out.to(images.device)
