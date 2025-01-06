import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess

def generate_launch_description():
    # RViz configuration file path
    rviz_config_path = os.path.join(
        get_package_share_directory('point_cloud_processor'),
        'rviz',
        'pointcloud_human_detection.rviz'
    )

    return LaunchDescription([
        # Launch the PointCloudProcessor Node
        Node(
            package='point_cloud_processor',
            executable='point_cloud_processor_node',
            name='point_cloud_processor',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
        
        # Launch the HumanClusterDetector Node
        Node(
            package='point_cloud_processor',
            executable='human_cluster_detector_node',
            name='human_cluster_detector',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),

        # Launch RViz2 with the fixed frame set to 'rslidar'
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_path],
            output='screen'
        )
    ])
