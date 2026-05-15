"""Allowlisted public defensive telemetry datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a known public defensive telemetry dataset."""

    key: str
    name: str
    homepage_url: str
    direct_download_url: str | None
    archive_type: str | None
    default_csv_glob: str
    description: str
    citation_hint: str
    requires_manual_download: bool = False
    notes: str = ""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="wustl-iiot-2021",
        name="WUSTL-IIoT-2021",
        homepage_url="https://research.engineering.wustl.edu/~jain/iiot2/index.html",
        direct_download_url="https://research.engineering.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip",
        archive_type="zip",
        default_csv_glob="*.csv",
        description="Industrial IoT cybersecurity dataset from the Washington University IIoT testbed.",
        citation_hint="Zolanvari et al., WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research.",
        notes=(
            "The official page warns that some identifier/time columns can leak labels in ML studies. "
            "Use source/destination fields for replay visibility, but avoid training production models on leakage-prone columns."
        ),
    ),
    DatasetSpec(
        key="ciciot2023",
        name="CICIoT2023",
        homepage_url="https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        direct_download_url=None,
        archive_type=None,
        default_csv_glob="**/*.csv",
        description="Large-scale IoT attack dataset from the Canadian Institute for Cybersecurity.",
        citation_hint="Neto et al., CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.",
        requires_manual_download=True,
        notes="The official download currently uses a CIC web form, so the runtime cannot fetch it unattended.",
    ),
    DatasetSpec(
        key="ton-iot",
        name="ToN_IoT",
        homepage_url="https://research.unsw.edu.au/projects/toniot-datasets",
        direct_download_url=None,
        archive_type=None,
        default_csv_glob="**/*.csv",
        description="UNSW IoT/IIoT telemetry, OS, and network datasets for AI-driven cybersecurity.",
        citation_hint="Moustafa et al., ToN_IoT dataset papers.",
        requires_manual_download=True,
        notes="The official download is hosted through UNSW SharePoint and may require interactive access.",
    ),
    DatasetSpec(
        key="bot-iot",
        name="BoT-IoT",
        homepage_url="https://research.unsw.edu.au/projects/bot-iot-dataset",
        direct_download_url=None,
        archive_type=None,
        default_csv_glob="**/*.csv",
        description="UNSW Cyber Range IoT botnet dataset with PCAP, Argus, and CSV artifacts.",
        citation_hint="Koroniotis et al., Towards the development of realistic botnet dataset in the Internet of Things.",
        requires_manual_download=True,
        notes="The official download is hosted through UNSW SharePoint and may require interactive access.",
    ),
    DatasetSpec(
        key="edge-iiotset",
        name="Edge-IIoTset",
        homepage_url="https://www.iotdataset.com/data/edge-iiotset-cyber-security-dataset",
        direct_download_url=None,
        archive_type=None,
        default_csv_glob="**/*.csv",
        description="Edge/IIoT security dataset with multiple IoT and IIoT attack categories.",
        citation_hint="Ferrag et al., Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset.",
        requires_manual_download=True,
        notes="Keep local copies outside git and pass the CSV path with replay-csv.",
    ),
)


def list_datasets() -> tuple[DatasetSpec, ...]:
    """Return known dataset specs."""

    return DATASETS


def dataset_by_key(key: str) -> DatasetSpec:
    """Look up a dataset by key."""

    normalized = key.strip().lower()
    for dataset in DATASETS:
        if dataset.key == normalized:
            return dataset
    known = ", ".join(dataset.key for dataset in DATASETS)
    raise KeyError(f"Unknown dataset '{key}'. Known datasets: {known}")

