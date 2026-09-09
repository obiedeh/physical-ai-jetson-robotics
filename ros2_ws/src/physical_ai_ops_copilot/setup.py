from setuptools import find_packages, setup

package_name = "physical_ai_ops_copilot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/ops_copilot.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Obinna Edeh",
    maintainer_email="obiedeh@gmail.com",
    description=(
        "ROS 2 agentic ops copilot node — deterministic triage and optional "
        "LLM enrichment over robot telemetry."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            f"ops_copilot_node = {package_name}.ops_copilot_node:main",
        ],
    },
)
