from physical_ai_lab.config import ComputeTarget
from physical_ai_lab.edge_ai import default_edge_benchmark_plan


def test_default_edge_benchmark_plan_has_measurable_workloads() -> None:
    plan = default_edge_benchmark_plan(ComputeTarget.jetson_orin)

    assert plan.compute_target == ComputeTarget.jetson_orin
    assert plan.evidence_path.endswith("jetson_orin_benchmarks.json")
    assert plan.readiness_score == 1.0
    assert all(workload.minimum_fps > 0 for workload in plan.workloads)
    assert all(workload.maximum_latency_ms > 0 for workload in plan.workloads)


def test_workstation_plan_targets_openusd_inputs() -> None:
    plan = default_edge_benchmark_plan(ComputeTarget.workstation)

    assert any("OpenUSD" in workload.input_source for workload in plan.workloads)
