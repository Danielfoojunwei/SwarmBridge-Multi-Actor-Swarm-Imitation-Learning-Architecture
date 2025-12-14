cmake_minimum_required(VERSION 3.8)
project(swarm_capture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# Camera capture node
add_executable(camera_capture_node
  src/camera_capture_node.cpp
)

ament_target_dependencies(camera_capture_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

install(TARGETS
  camera_capture_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
