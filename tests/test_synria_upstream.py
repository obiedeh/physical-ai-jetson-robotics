from physical_ai_lab.synria_upstream import (
    ActivityPhase,
    synria_upstream_activities,
    synria_upstream_activity_by_id,
)


def test_synria_upstream_activities_include_rtx_and_hardware_phases() -> None:
    activities = synria_upstream_activities()
    phases = {activity.phase for activity in activities}

    assert ActivityPhase.rtx_now in phases
    assert ActivityPhase.hardware_later in phases
    assert len(activities) >= 10
    assert all(activity.source_repo for activity in activities)
    assert all(activity.deliverable for activity in activities)


def test_synria_upstream_activity_lookup_by_id() -> None:
    activity = synria_upstream_activity_by_id("c10-camera-sim-contract")

    assert activity.phase == ActivityPhase.rtx_now
    assert "C10" in activity.title
    assert "Synria-C10-SDK" in activity.source_repo
