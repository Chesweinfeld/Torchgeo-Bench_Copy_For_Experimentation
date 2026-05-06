"""Tests for the high-level get_datasets API for GeoBench V2 datasets."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets


class MockV2Dataset:
    """Stand-in for ``geobench_v2.datasets.GeoBench<X>`` upstream classes."""

    def __init__(self, root, split, transforms=None, band_order=None, **kwargs):
        del kwargs
        self.root = root
        self.split = split
        self.transforms = transforms
        self.band_order = band_order

        if isinstance(band_order, dict):
            self.c = sum(len(v) for v in band_order.values())
        elif band_order is not None:
            self.c = len(band_order)
        else:
            self.c = 3
        self.h, self.w = 32, 32

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        img = torch.randn(self.c, self.h, self.w)
        sample = {"image": img}
        if self.transforms:
            sample = self.transforms(sample)
        # Return both label and mask so segmentation+classification tests both pass.
        sample.setdefault("label", torch.tensor(1))
        sample.setdefault("mask", torch.randint(0, 2, (self.h, self.w)))
        return sample


@pytest.fixture
def mock_v2_env():
    """Patch the V2 dataset classes at their upstream module location."""
    with (
        patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ),
        patch(
            "geobench_v2.datasets.GeoBenchBurnScars",
            MagicMock(side_effect=MockV2Dataset),
        ),
        patch(
            "geobench_v2.datasets.GeoBenchSo2Sat",
            MagicMock(side_effect=MockV2Dataset),
        ),
    ):
        yield


class TestV2Loading:
    def test_benv2_classification(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, val_dl, test_dl = get_datasets(
            dataset_name="benv2",
            return_val=True,
            batch_size=4,
            num_workers=0,
        )

        assert isinstance(train_dl, DataLoader)
        assert len(ds) == 10
        assert get_bench_dataset_class("benv2").task == "classification"

        batch = next(iter(train_dl))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert "label" in batch

    def test_burn_scars_segmentation(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, test_dl = get_datasets(
            dataset_name="burn_scars",
            batch_size=2,
            return_val=False,
            num_workers=0,
        )

        assert get_bench_dataset_class("burn_scars").task == "segmentation"
        batch = next(iter(train_dl))
        assert "mask" in batch
        assert batch["image"].shape[0] == 2

    def test_partition_warning(self, mock_v2_env):
        del mock_v2_env
        with pytest.warns(UserWarning, match="does not support custom partitions"):
            get_datasets(
                dataset_name="benv2",
                partition_name="0.10x_train",
                num_workers=0,
            )

    def test_resize_transform(self, mock_v2_env):
        del mock_v2_env
        target = 64
        ds, _, _ = get_datasets(
            dataset_name="benv2",
            image_size=target,
            batch_size=4,
            num_workers=0,
        )

        # The resize transform is forwarded as ``transforms`` to the upstream
        # mock through ``GeoBenchv2._inner``.
        assert ds._inner.transforms is not None

        dl = DataLoader(ds, batch_size=1)
        batch = next(iter(dl))
        assert batch["image"].shape[-1] == target

    def test_bad_dataset_name(self):
        with pytest.raises(KeyError, match="Unknown dataset 'phantom_dataset'"):
            get_datasets(dataset_name="phantom_dataset")

    def test_no_double_root_join(self, mock_v2_env):
        """``GeoBenchv2`` must combine collection-root + dataset-name once."""
        with patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="benv2",
                batch_size=2,
                num_workers=0,
            )
            assert mocked.call_count == 3  # train, val, test
            for call in mocked.call_args_list:
                kwargs = call.kwargs
                assert Path(kwargs["root"]) == Path("data/geobenchv2/benv2"), kwargs

    def test_band_order_shape_dict(self, mock_v2_env):
        """Multi-modality V2 wrappers must hand a dict ``band_order`` upstream."""
        with patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="benv2",
                bands="rgb",
                batch_size=2,
                num_workers=0,
            )
            for call in mocked.call_args_list:
                bo = call.kwargs["band_order"]
                assert isinstance(bo, dict), bo
                assert bo == {"s2": ["B04", "B03", "B02"]}, bo

    def test_band_order_shape_flat(self, mock_v2_env):
        """Single-modality V2 wrappers must hand a flat list ``band_order`` upstream."""
        with patch(
            "geobench_v2.datasets.GeoBenchBurnScars",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="burn_scars",
                bands="rgb",
                batch_size=2,
                num_workers=0,
            )
            for call in mocked.call_args_list:
                bo = call.kwargs["band_order"]
                assert isinstance(bo, list), bo
                assert bo == ["B04", "B03", "B02"], bo
