"""Download and prepare allowlisted public telemetry datasets."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from jetson_edge_ai_security.datasets.catalog import DatasetSpec, dataset_by_key


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or prepared."""


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared local dataset paths."""

    spec: DatasetSpec
    cache_dir: Path
    archive_path: Path
    extract_dir: Path
    csv_path: Path


def prepare_dataset(
    key: str,
    *,
    cache_dir: str | Path = Path("data/datasets"),
    force: bool = False,
    csv_glob: str | None = None,
) -> PreparedDataset:
    """Download, extract, and locate a CSV for a known dataset."""

    spec = dataset_by_key(key)
    if spec.requires_manual_download or spec.direct_download_url is None:
        raise DatasetDownloadError(
            f"{spec.name} does not have an unattended direct download URL. "
            f"Download it from {spec.homepage_url}, then use replay-csv --path <local.csv>."
        )

    root = Path(cache_dir)
    dataset_dir = root / spec.key
    archive_path = dataset_dir / _archive_name(spec)
    extract_dir = dataset_dir / "extracted"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if force and dataset_dir.exists():
        if not dataset_dir.resolve().is_relative_to(root.resolve()):
            raise DatasetDownloadError(f"Refusing to clear dataset path outside cache root: {dataset_dir}")
        shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        _download(spec.direct_download_url, archive_path)
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extract_dir, spec.archive_type)

    csv_path = _find_csv(extract_dir, csv_glob or spec.default_csv_glob)
    return PreparedDataset(
        spec=spec,
        cache_dir=dataset_dir,
        archive_path=archive_path,
        extract_dir=extract_dir,
        csv_path=csv_path,
    )


def _archive_name(spec: DatasetSpec) -> str:
    parsed = urlparse(spec.direct_download_url or "")
    name = Path(parsed.path).name
    return name or f"{spec.key}.{spec.archive_type or 'archive'}"


def _download(url: str, destination: Path) -> None:
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:  # pragma: no cover - exact urllib errors vary by platform
        raise DatasetDownloadError(f"Failed to download {url}: {exc}") from exc


def _extract_archive(archive_path: Path, extract_dir: Path, archive_type: str | None) -> None:
    if archive_type == "zip" or archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _safe_extract_path(extract_dir, member.filename)
            archive.extractall(extract_dir)
        return

    if archive_type in {"tar", "tgz", "tar.gz"} or archive_path.suffix.lower() in {".tar", ".gz", ".tgz"}:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                _safe_extract_path(extract_dir, member.name)
            archive.extractall(extract_dir)
        return

    if archive_path.suffix.lower() == ".csv":
        shutil.copy2(archive_path, extract_dir / archive_path.name)
        return

    raise DatasetDownloadError(f"Unsupported archive type for {archive_path}")


def _safe_extract_path(extract_dir: Path, member_name: str) -> None:
    target = (extract_dir / member_name).resolve()
    root = extract_dir.resolve()
    if not target.is_relative_to(root):
        raise DatasetDownloadError(f"Archive member would extract outside target directory: {member_name}")


def _find_csv(extract_dir: Path, pattern: str) -> Path:
    candidates = [path for path in extract_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".csv"]
    if not candidates:
        candidates = [path for path in extract_dir.rglob("*.csv") if path.is_file()]
    if not candidates:
        raise DatasetDownloadError(f"No CSV files found under {extract_dir}")
    return max(candidates, key=lambda path: path.stat().st_size)

