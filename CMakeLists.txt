cmake_minimum_required(VERSION 3.5)
project(point_cloud_processor)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Add executables for both files
add_executable(point_cloud_processor_node src/point_cloud_processor_node.cpp)
ament_target_dependencies(point_cloud_processor_node rclcpp pcl_ros sensor_msgs)
target_link_libraries(point_cloud_processor_node ${PCL_LIBRARIES})

add_executable(human_cluster_detector_node src/human_cluster_detector_node.cpp)
ament_target_dependencies(human_cluster_detector_node rclcpp pcl_ros sensor_msgs)
target_link_libraries(human_cluster_detector_node ${PCL_LIBRARIES})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  USE_SOURCE_PERMISSIONS
)
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# Install targets
install(TARGETS
  point_cloud_processor_node
  human_cluster_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
