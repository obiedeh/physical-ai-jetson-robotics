# Public Datasets

The runtime supports two dataset modes:

```text
local CSV path -> replay-csv
known dataset key -> fetch-dataset/replay-dataset
```

`fetch-dataset` only downloads from allowlisted direct URLs. This keeps the runtime predictable and avoids scraping forms, executing downloaded code, or guessing which file to use from opaque portals.

## Supported Auto-Download

| Key | Dataset | Source | Notes |
|---|---|---|---|
| `wustl-iiot-2021` | WUSTL-IIoT-2021 | `https://research.engineering.wustl.edu/~jain/iiot2/index.html` | Official ZIP download; useful IIoT testbed benchmark. |

Example:

```bash
edge-security fetch-dataset wustl-iiot-2021
edge-security replay-dataset wustl-iiot-2021 --limit 1000
```

## Manual Download Recommended

| Key | Dataset | Source | Why Manual |
|---|---|---|---|
| `ciciot2023` | CICIoT2023 | `https://www.unb.ca/cic/datasets/iotdataset-2023.html` | Official download currently uses a CIC form. |
| `ton-iot` | ToN_IoT | `https://research.unsw.edu.au/projects/toniot-datasets` | Official download is hosted through UNSW SharePoint. |
| `bot-iot` | BoT-IoT | `https://research.unsw.edu.au/projects/bot-iot-dataset` | Official download is hosted through UNSW SharePoint. |
| `edge-iiotset` | Edge-IIoTset | `https://www.iotdataset.com/data/edge-iiotset-cyber-security-dataset` | Keep local copies outside git and replay selected CSVs. |

For manual datasets:

```bash
edge-security replay-csv --path data/datasets/<dataset>/<file>.csv --limit 1000
```

## Safety Notes

- Archives are extracted with path traversal checks.
- Downloaded code or notebooks are never executed.
- The runtime chooses the largest CSV by default unless `--csv-glob` is provided.
- Keep large datasets out of git.
- Treat all attack samples as defensive replay/lab telemetry only.

## Leakage Notes

Some datasets include columns that can leak the answer during ML training. WUSTL-IIoT-2021, for example, warns about identifier/time columns on its official page. For runtime replay, these fields can help preserve flow visibility, but model training should use a reviewed feature list that excludes leakage-prone identifiers.

