import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import (
    FindPackageShare,
)

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    usd_file_name = 'go1_slam/resources/go1/go1.urdf'
    urdf = os.path.join(
        get_package_share_directory('go1_slam'),
        usd_file_name)
    with open(urdf, 'r') as info:
        robot_desc = info.read()

    return LaunchDescription([
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py',
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'slam_params_file': os.path.join(
                    get_package_share_directory('go1_slam'),
                    'config',
                    'mapper_params_online_async.yaml'
                ),
            }.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'use_sim_time': use_sim_time, 
                    'robot_description': robot_desc
                }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('go1_slam'), 'rviz2', 'config.rviz')]
        ),
        
    ])