#!/bin/bash
set -e

# Source ROS 2 installation
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace if built
if [ -f /opt/ros2_ws/install/setup.bash ]; then
    source /opt/ros2_ws/install/setup.bash
fi

# Execute command
exec "$@"
