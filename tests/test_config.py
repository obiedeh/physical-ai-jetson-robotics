from physical_ai_lab.config import ComputeTarget, LabProfile, RobotPlatform


def test_lab_profile_summary_mentions_compute_and_safety_gate() -> None:
    profile = LabProfile(
        name="jetson-smoke",
        compute_target=ComputeTarget.jetson_orin,
        robot_platforms=[RobotPlatform.robo_car],
        ros_domain_id=11,
    )

    summary = profile.summary()

    assert "jetson-smoke" in summary
    assert "jetson-orin" in summary
    assert "robo-car" in summary
    assert "safety_gate=enabled" in summary
