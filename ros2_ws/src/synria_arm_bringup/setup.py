from setuptools import find_packages, setup

package_name = "synria_arm_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/leader_state.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Obinna Edeh",
    maintainer_email="obiedeh@gmail.com",
    description="Read-only leader state bring-up for Synria data recording.",
    license="MIT",
)
