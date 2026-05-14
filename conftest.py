import pytest


def pytest_configure(config: pytest.Config) -> None:
    # ROS 2 installs a launch_testing_ros pytest plugin that registers
    # a non-standard hook (pytest_launch_collect_makemodule). On machines
    # where lark is missing or the hook spec doesn't match, pytest raises
    # PluginValidationError before any tests run. Unregister it here so
    # the plain Python test suite can run without a ROS environment.
    for name in ("launch-testing-ros", "launch_testing_ros"):
        plugin = config.pluginmanager.get_plugin(name)
        if plugin is not None:
            config.pluginmanager.unregister(plugin)
