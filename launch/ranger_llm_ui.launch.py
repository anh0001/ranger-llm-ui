"""
ROS 2 Launch file for Ranger LLM UI.

This launch file starts the Ranger LLM UI node with configurable parameters.

Usage:
    ros2 launch ranger_llm_ui ranger_llm_ui.launch.py
    ros2 launch ranger_llm_ui ranger_llm_ui.launch.py llm_provider:=ollama
    ros2 launch ranger_llm_ui ranger_llm_ui.launch.py simple_mode:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Ranger LLM UI."""

    # Declare launch arguments
    llm_provider_arg = DeclareLaunchArgument(
        'llm_provider',
        default_value=EnvironmentVariable('LLM_PROVIDER', default_value='openai'),
        description='LLM provider: openai, ollama, or anthropic'
    )

    llm_model_arg = DeclareLaunchArgument(
        'llm_model',
        default_value='',
        description='LLM model name (empty for provider default)'
    )

    gradio_port_arg = DeclareLaunchArgument(
        'gradio_port',
        default_value='7860',
        description='Gradio web server port'
    )

    share_arg = DeclareLaunchArgument(
        'share',
        default_value='false',
        description='Create a public Gradio share link'
    )

    simple_mode_arg = DeclareLaunchArgument(
        'simple_mode',
        default_value='false',
        description='Use simple agent without LLM (for testing)'
    )

    # Get launch configurations
    llm_provider = LaunchConfiguration('llm_provider')
    llm_model = LaunchConfiguration('llm_model')
    gradio_port = LaunchConfiguration('gradio_port')
    share = LaunchConfiguration('share')
    simple_mode = LaunchConfiguration('simple_mode')

    # Build command with arguments
    # Note: We use ExecuteProcess because we need to pass command-line args
    # to the Python script in a way that ROS 2 node doesn't directly support
    ui_node = ExecuteProcess(
        cmd=[
            'python3', '-m', 'ranger_llm_ui.ui_node',
            '--provider', llm_provider,
            '--model', llm_model,
            '--port', gradio_port,
        ],
        name='ranger_llm_ui',
        output='screen',
    )

    return LaunchDescription([
        llm_provider_arg,
        llm_model_arg,
        gradio_port_arg,
        share_arg,
        simple_mode_arg,
        ui_node,
    ])
