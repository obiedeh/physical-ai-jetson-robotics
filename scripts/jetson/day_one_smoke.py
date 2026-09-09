#!/usr/bin/env python3
"""Jetson day-one telemetry smoke — the first REAL measurement on a new
board, recorded with the audit-grade discipline in
physical_ai_lab.jetson_provenance (provenance BEFORE measuring, explicit
data_kind, seeds, n/p50/p95/p99/std + warm/cold on every latency,
sysfs thermal + power sampling — never tegrastats-with-timeout).

Probes (each optional, each labeled unavailable rather than skipped):
  1. CUDA fp16 matmul latency, cold (first call) + warm (n=200)
  2. camera grab via V4L2 (cv2), frame interval over 60 frames
  3. INA3221 VDD_IN power (sysfs) + thermal zones across the run

Run ON THE BOARD from the repo root with the system python (Jetson torch
wheels live there, not in .venv):

    PYTHONPATH=. python3 scripts/jetson/day_one_smoke.py \
        [--out reports/jetson/<host>_day_one] [--camera 0] [--n 200]
"""
from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import threading
import time
from pathlib import Path

from physical_ai_lab.jetson_provenance import (
    ThermalWatch, fix_seeds, latency_stats, write_provenance,
)


class PowerWatch:
    """INA3221 rail sampler (sysfs). mW = mV * mA / 1000."""

    def __init__(self, interval_s: float = 0.5):
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
        self._th = None

    def _read(self):
        s = {"t": time.time()}
        for name, (vp, cp) in self.rails.items():
            try:
                mv = int(open(vp).read()); ma = int(open(cp).read())
                s[name + "_mW"] = round(mv * ma / 1000.0, 1)
            except OSError:
                pass
        return s

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._read())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
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


def _power_mode_nosudo() -> str:
    try:
        status = open("/var/lib/nvpmodel/status").read().strip()  # pmode:0000
        pid = int(status.split(":")[1])
        for line in open("/etc/nvpmodel.conf"):
            if line.startswith(f"< POWER_MODEL ID={pid} "):
                return f"{line.strip()} ({status})"
        return status
    except Exception:
        return "unavailable"


def probe_matmul(n: int, size: int) -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda unavailable"}
    dev = torch.device("cuda")
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); (a @ b); torch.cuda.synchronize()
    cold = [(time.perf_counter() - t0) * 1000.0]
    for _ in range(10):
        (a @ b)
    torch.cuda.synchronize()
    warm = []
    for _ in range(n):
        t0 = time.perf_counter(); (a @ b); torch.cuda.synchronize()
        warm.append((time.perf_counter() - t0) * 1000.0)
    flops = 2.0 * size ** 3
    ws = latency_stats(warm, warm=True, label=f"matmul_fp16_{size}")
    return {"available": True, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "size": size,
            "cold_first_call_ms": round(cold[0], 3), "warm": ws,
            "warm_tflops_p50": round(flops / (ws["p50_ms"] / 1000.0) / 1e12, 2),
            "mem_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1)}


def probe_camera(index: int, frames: int) -> dict:
    try:
        import cv2
    except ImportError:
        return {"available": False, "reason": "cv2 not importable"}
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return {"available": False, "reason": f"/dev/video{index} failed to open"}
    ok0, _ = cap.read()
    if not ok0:
        cap.release()
        return {"available": False, "reason": "opened but no frame"}
    ts = []
    t_prev = time.perf_counter()
    got = 0
    for _ in range(frames):
        ok, fr = cap.read()
        if not ok:
            break
        t = time.perf_counter(); ts.append((t - t_prev) * 1000.0); t_prev = t
        got += 1
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    out = {"available": True, "device": f"/dev/video{index}",
           "frames": got, "width": int(w), "height": int(h),
           "cv2": cv2.__version__}
    if got >= 2:
        out["frame_interval"] = latency_stats(ts, warm=True,
                                              label=f"v4l2_grab_video{index}")
        out["fps_from_p50"] = round(1000.0 / out["frame_interval"]["p50_ms"], 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    host = platform.node()
    out = args.out or Path("reports/jetson") / f"{host}_day_one"
    out.mkdir(parents=True, exist_ok=True)
    seed_rec = fix_seeds(args.seed)
    model = "unavailable"
    try:
        model = open("/proc/device-tree/model").read().replace("\0", "").strip()
    except OSError:
        pass
    prov = write_provenance(
        out, data_kind=(f"REAL hardware measurement ({model}; on-device "
                        "CUDA matmul + V4L2 camera grab; NOT simulated)"),
        seed_record=seed_rec, precision="fp16", batch_size=1,
        extra={"power_mode_nosudo": _power_mode_nosudo(),
               "timezone": time.strftime("%Z %z"),
               "probe_plan": ["matmul_fp16", f"camera_video{args.camera}",
                              "ina3221_power", "thermal_zones"]})
    print(f"[day-one] provenance -> {prov}", flush=True)
    res = {"host": host, "device_model": model,
           "provenance": prov.name, "probes": {}}
    with ThermalWatch(interval_s=1.0) as tw, PowerWatch(0.5) as pw:
        idle_t0 = time.time()
        time.sleep(3.0)                       # idle baseline window
        idle_n = len(pw.samples)
        res["probes"]["matmul"] = probe_matmul(args.n, args.size)
        print(f"[day-one] matmul: {json.dumps(res['probes']['matmul'])[:200]}",
              flush=True)
        res["probes"]["camera"] = probe_camera(args.camera, 60)
        print(f"[day-one] camera: {json.dumps(res['probes']['camera'])[:200]}",
              flush=True)
        time.sleep(1.0)
    res["thermal"] = tw.summary()
    res["power"] = pw.summary()
    idle = [s for s in pw.samples[:idle_n]]
    for rail in pw.rails:
        v = sorted(s[rail + "_mW"] for s in idle if rail + "_mW" in s)
        if v:
            res["power"].setdefault("idle_baseline", {})[rail] = {
                "mW_p50": v[len(v) // 2], "n": len(v),
                "window_s": round(3.0, 1)}
    res["wall_clock_s"] = round(time.time() - idle_t0, 1)
    p = out / "day_one_smoke.json"
    i = 1
    while p.exists():
        p = out / f"day_one_smoke.{i}.json"; i += 1
    p.write_text(json.dumps(res, indent=2))
    print(f"[day-one] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
