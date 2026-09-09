"""Run-provenance writer for Jetson-class hardware (Yahboom Orin SLAM
platform and any future embedded target). Port of the executor's
provenance discipline, extended with every field the 2026-08-19
portfolio audit found missing from all existing Jetson artifacts:
power mode, precision label, batch size, warm/cold, throttling state,
sustained-vs-peak memory, and a variance figure required alongside
every latency.

Design rules (audit lessons, non-negotiable):
1. Call ``write_provenance()`` BEFORE the first measurement of any run.
2. ``data_kind`` must be the explicit POSITIVE form on hardware, e.g.
   "REAL hardware measurement (Jetson Orin <module>, live sensors)" —
   never an implicit default; sim stays loudly labeled sim.
3. Seeds: ``fix_seeds()`` seeds python/numpy/torch AND sets
   PYTHONHASHSEED + CUDA determinism flags, and the values land in the
   provenance dict. (Two portfolio repos had zero seed occurrences.)
4. Latency reporting contract: use ``latency_stats()`` — it refuses to
   emit a bare mean; every latency carries n, p50/p95/p99, mean, std,
   and warm/cold labeling.
5. Thermal/power: use ``ThermalWatch`` (sysfs polling in a thread) —
   NEVER ``subprocess.run(["tegrastats"], timeout=...)``, which streams
   forever, always raises TimeoutExpired, and yields null power on
   every run (the jetson-edge-ai-security bug, audit E1).
"""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


def _cmd(c: str) -> str:
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """round((p/100)*(n-1)) indexing — the portfolio's one correct
    percentile implementation (safety-observability convention)."""
    if not sorted_vals:
        return None
    return sorted_vals[round((p / 100.0) * (len(sorted_vals) - 1))]


def fix_seeds(seed: int) -> dict:
    """Seed everything, return the record for provenance."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    rec = {"seed": seed, "PYTHONHASHSEED": str(seed)}
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        rec["numpy_seeded"] = True
    except ImportError:
        rec["numpy_seeded"] = False
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        rec["torch_seeded"] = True
        rec["cuda_determinism"] = "use_deterministic_algorithms(warn_only)"
        rec["cudnn_deterministic"] = True
    except ImportError:
        rec["torch_seeded"] = False
    return rec


def jetson_facts() -> dict:
    """Everything knowable about the board, machine-read, no prose."""
    facts = {
        "device_model": _cmd("cat /proc/device-tree/model 2>/dev/null"
                             " | tr -d '\\0'") or "unavailable",
        "l4t_release": _cmd("cat /etc/nv_tegra_release | head -1"),
        "jetpack_apt": _cmd("dpkg -l 2>/dev/null | grep -m1 nvidia-jetpack"
                            " || true"),
        "power_mode": _cmd("nvpmodel -q 2>/dev/null | head -2"
                           " || sudo -n nvpmodel -q 2>/dev/null | head -2"),
        "jetson_clocks": _cmd("jetson_clocks --show 2>/dev/null | head -3"
                              " || true"),
        "cuda": _cmd("nvcc --version 2>/dev/null | grep release || true"),
        "tensorrt": _cmd("dpkg -l 2>/dev/null | grep -m1 'libnvinfer[0-9]'"
                         " || python3 -c 'import tensorrt;"
                         "print(tensorrt.__version__)' 2>/dev/null || true"),
    }
    return facts


def write_provenance(out_dir: Path, *, data_kind: str, seed_record: dict,
                     precision: str, batch_size: int,
                     extra: dict | None = None) -> Path:
    """Append-only provenance, written BEFORE the first measurement.

    data_kind is REQUIRED and must be explicit — for hardware use the
    positive form: "REAL hardware measurement (<device>, <sensors>)".
    """
    if "real" not in data_kind.lower() and "sim" not in data_kind.lower() \
            and "synthetic" not in data_kind.lower() \
            and "fixture" not in data_kind.lower():
        raise ValueError(
            "data_kind must state real/simulated/synthetic/fixture "
            "explicitly (audit rule).")
    prov = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _cmd("git rev-parse HEAD"),
        "git_dirty": bool(_cmd("git status --porcelain")),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "data_kind": data_kind,
        "precision": precision,
        "batch_size": batch_size,
        **jetson_facts(),
        **seed_record,
    }
    if prov["git_dirty"]:
        import hashlib
        diff = _cmd("git diff HEAD")
        patch = out_dir / "uncommitted.patch"
        i = 1
        while patch.exists():
            patch = out_dir / f"uncommitted.{i}.patch"
            i += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        patch.write_text(diff)
        prov["uncommitted_patch"] = patch.name
        prov["uncommitted_patch_sha256"] = hashlib.sha256(
            diff.encode()).hexdigest()
    if extra:
        prov.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "provenance.json"
    i = 1
    while p.exists():
        p = out_dir / f"provenance.{i}.json"
        i += 1
    p.write_text(json.dumps(prov, indent=2))
    return p


def latency_stats(samples_ms: list, *, warm: bool, label: str) -> dict:
    """A latency is only reportable with n, percentiles, variance and a
    warm/cold label. Refuses bare means by construction."""
    if len(samples_ms) < 2:
        raise ValueError(f"{label}: need >=2 samples for a variance figure "
                         f"(got {len(samples_ms)}); a bare mean is not "
                         "reportable (audit rule 2).")
    s = sorted(samples_ms)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / (n - 1)
    return {"label": label, "n": n, "warm": warm,
            "p50_ms": _percentile(s, 50), "p95_ms": _percentile(s, 95),
            "p99_ms": _percentile(s, 99), "mean_ms": round(mean, 4),
            "std_ms": round(var ** 0.5, 4),
            "min_ms": s[0], "max_ms": s[-1]}


class ThermalWatch:
    """sysfs thermal + RAM sampler in a thread; summarizes peak vs
    sustained and flags throttling. Use as a context manager around the
    measured region. Never uses subprocess timeouts against streaming
    tools."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _zones(self) -> dict[str, float]:
        import glob
        out = {}
        for z in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                out[open(z + "/type").read().strip()] = \
                    int(open(z + "/temp").read()) / 1000.0
            except OSError:
                pass
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            s = {"t": time.time(), **self._zones()}
            try:
                mem = open("/proc/meminfo").read()
                s["mem_available_kb"] = int(
                    mem.split("MemAvailable:")[1].split()[0])
            except Exception:
                pass
            self.samples.append(s)
            self._stop.wait(self.interval)

    def __enter__(self) -> ThermalWatch:
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def summary(self, throttle_at_c: float = 90.0) -> dict:
        temps = [max(v for k, v in s.items()
                     if isinstance(v, float) and k != "t")
                 for s in self.samples if len(s) > 2]
        temps_sorted = sorted(temps)
        mems = [s["mem_available_kb"] for s in self.samples
                if "mem_available_kb" in s]
        return {
            "samples": len(self.samples),
            "temp_c_peak": max(temps) if temps else None,
            "temp_c_sustained_p50": _percentile(temps_sorted, 50),
            "throttling_suspected": bool(temps) and max(temps) >= throttle_at_c,
            "throttle_threshold_c": throttle_at_c,
            "mem_available_kb_min": min(mems) if mems else None,
            "mem_available_kb_p50": _percentile(sorted(mems), 50)
            if mems else None,
        }


class PowerWatch:
    """INA3221 rail sampler (sysfs). mW = mV * mA / 1000."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval = interval_s
        self.samples: list[dict] = []
        self.rails: dict[str, tuple[str, str]] = {}
        for hw in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
            for lab in glob.glob(hw + "/in*_label"):
                idx = Path(lab).name[2:-6]
                try:
                    name = open(lab).read().strip()
                except OSError:
                    continue
                self.rails[name] = (f"{hw}/in{idx}_input",
                                    f"{hw}/curr{idx}_input")
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _read(self) -> dict:
        s: dict = {"t": time.time()}
        for name, (vp, cp) in self.rails.items():
            try:
                mv = int(open(vp).read())
                ma = int(open(cp).read())
                s[name + "_mW"] = round(mv * ma / 1000.0, 1)
            except OSError:
                pass
        return s

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._read())
            self._stop.wait(self.interval)

    def __enter__(self) -> PowerWatch:
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def summary(self) -> dict:
        out = {"rails": sorted(self.rails), "samples": len(self.samples)}
        for name in self.rails:
            vals = sorted(s[name + "_mW"] for s in self.samples
                          if name + "_mW" in s)
            if vals:
                out[name] = {"mW_peak": vals[-1],
                             "mW_p50": vals[len(vals) // 2],
                             "mW_min": vals[0], "n": len(vals)}
        return out
