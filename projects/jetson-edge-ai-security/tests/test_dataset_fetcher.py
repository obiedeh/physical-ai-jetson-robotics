from pathlib import Path
from zipfile import ZipFile

import pytest

from jetson_edge_ai_security.datasets.catalog import DatasetSpec
from jetson_edge_ai_security.datasets.fetcher import DatasetDownloadError, prepare_dataset
import jetson_edge_ai_security.datasets.fetcher as fetcher


def test_prepare_dataset_downloads_extracts_and_finds_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "nested/events.csv",
            "timestamp,source_ip,dest_ip,protocol,packet_size\n"
            "2026-01-01 00:00:00,10.0.0.1,10.0.0.2,TCP,128\n",
        )

    spec = DatasetSpec(
        key="fixture",
        name="Fixture",
        homepage_url="https://example.test/fixture",
        direct_download_url=archive_path.as_uri(),
        archive_type="zip",
        default_csv_glob="**/*.csv",
        description="Test fixture",
        citation_hint="Test fixture",
    )
    monkeypatch.setattr(fetcher, "dataset_by_key", lambda key: spec)

    prepared = prepare_dataset("fixture", cache_dir=tmp_path / "cache")

    assert prepared.csv_path.name == "events.csv"
    assert prepared.csv_path.read_text(encoding="utf-8").startswith("timestamp")


def test_prepare_dataset_rejects_manual_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = DatasetSpec(
        key="manual",
        name="Manual",
        homepage_url="https://example.test/manual",
        direct_download_url=None,
        archive_type=None,
        default_csv_glob="**/*.csv",
        description="Manual fixture",
        citation_hint="Manual fixture",
        requires_manual_download=True,
    )
    monkeypatch.setattr(fetcher, "dataset_by_key", lambda key: spec)

    with pytest.raises(DatasetDownloadError):
        prepare_dataset("manual", cache_dir=tmp_path / "cache")

