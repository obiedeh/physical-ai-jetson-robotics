cmake_minimum_required(VERSION 3.8)
project(rosmaster_m3pro_perception)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
