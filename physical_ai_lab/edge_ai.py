"""Edge AI benchmark planning for Jetson robotics workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from physical_ai_lab.config import ComputeTarget


@dataclass(frozen=True)
class EdgeBenchmarkTarget:
    """One measurable inference target for edge deployment validation."""

    name: str
    model_format: str
    input_source: str
    minimum_fps: float
    maximum_latency_ms: float
    maximum_power_w: float
    maximum_memory_mb: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeBenchmarkPlan:
    """A practical benchmark plan for one compute target."""

    compute_target: ComputeTarget
    workloads: list[EdgeBenchmarkTarget]
    evidence_path: str

    @property
    def readiness_score(self) -> float:
        if not self.workloads:
            return 0.0
        # A plan is "ready" when every workload has an explicit latency, FPS,
        # memory, and power target that can be measured on hardware.
        measurable = sum(
            1
            for workload in self.workloads
            if workload.minimum_fps > 0
            and workload.maximum_latency_ms > 0
            and workload.maximum_power_w > 0
            and workload.maximum_memory_mb > 0
        )
        return round(measurable / len(self.workloads), 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "compute_target": self.compute_target.value,
            "evidence_path": self.evidence_path,
            "readiness_score": self.readiness_score,
            "workloads": [workload.to_dict() for workload in self.workloads],
        }


def default_edge_benchmark_plan(compute_target: ComputeTarget) -> EdgeBenchmarkPlan:
    """Return the initial Jetson/workstation inference benchmark plan."""
    if compute_target == ComputeTarget.jetson_thor:
        return EdgeBenchmarkPlan(
            compute_target=compute_target,
            evidence_path="reports/edge_ai/jetson_thor_benchmarks.json",
            workloads=[
                EdgeBenchmarkTarget(
                    name="factory-cell-object-detection",
                    model_format="onnx/tensorrt",
                    input_source="C10 wrist camera or synthetic Isaac camera frames",
                    minimum_fps=45.0,
                    maximum_latency_ms=22.0,
                    maximum_power_w=65.0,
                    maximum_memory_mb=4096,
                ),
                EdgeBenchmarkTarget(
                    name="mobile-base-localization-assist",
                    model_format="onnx/tensorrt",
                    input_source="front RGB camera stream",
                    minimum_fps=30.0,
                    maximum_latency_ms=35.0,
                    maximum_power_w=70.0,
                    maximum_memory_mb=6144,
                ),
            ],
        )

    if compute_target == ComputeTarget.jetson_orin:
        return EdgeBenchmarkPlan(
            compute_target=compute_target,
            evidence_path="reports/edge_ai/jetson_orin_benchmarks.json",
            workloads=[
                EdgeBenchmarkTarget(
                    name="factory-cell-object-detection",
                    model_format="onnx/tensorrt",
                    input_source="C10 wrist camera or Yahboom camera stream",
                    minimum_fps=20.0,
                    maximum_latency_ms=50.0,
                    maximum_power_w=25.0,
                    maximum_memory_mb=3072,
                ),
                EdgeBenchmarkTarget(
                    name="operator-presence-safety-check",
                    model_format="onnx/tensorrt",
                    input_source="wide-angle workcell camera",
                    minimum_fps=15.0,
                    maximum_latency_ms=65.0,
                    maximum_power_w=25.0,
                    maximum_memory_mb=2048,
                ),
            ],
        )

    return EdgeBenchmarkPlan(
        compute_target=compute_target,
        evidence_path="reports/edge_ai/linux_rtx_benchmarks.json",
        workloads=[
            EdgeBenchmarkTarget(
                name="synthetic-data-label-check",
                model_format="onnx",
                input_source="OpenUSD rendered camera frames",
                minimum_fps=60.0,
                maximum_latency_ms=16.7,
                maximum_power_w=450.0,
                maximum_memory_mb=8192,
            ),
            EdgeBenchmarkTarget(
                name="policy-evaluation-batch",
                model_format="pytorch/onnx",
                input_source="recorded robot observation batches",
                minimum_fps=120.0,
                maximum_latency_ms=8.5,
                maximum_power_w=500.0,
                maximum_memory_mb=12288,
            ),
        ],
    )
