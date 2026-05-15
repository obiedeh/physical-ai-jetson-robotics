from physical_ai_lab.rtx_projects import high_impact_rtx_projects, rtx_project_by_id


def test_high_impact_rtx_projects_have_commands_and_evidence() -> None:
    projects = high_impact_rtx_projects()

    assert len(projects) >= 5
    assert all(project.command for project in projects)
    assert all(project.evidence for project in projects)
    assert any(project.identifier == "synria-reach-policy" for project in projects)


def test_rtx_project_lookup_by_id() -> None:
    project = rtx_project_by_id("ros-gazebo-factory-cell")

    assert project.title.startswith("ROS 2")
    assert "run_ros_gazebo_factory_cell.sh" in project.command
