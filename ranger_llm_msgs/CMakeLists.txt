cmake_minimum_required(VERSION 3.8)
project(ranger_llm_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/DriveDistance.action"
  "action/RotateAngle.action"
)

ament_package()
