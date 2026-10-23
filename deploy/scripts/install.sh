#!/usr/bin/env bash
set -euo pipefail
deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(dirname -- "$deploy_dir")"
# Source the installed ROS environment before running this script.
# System site packages expose distro-provided rclpy in the deployment venv.
# Explicit distro interpreter avoids an active Conda Python incompatible with Foxy rclpy.
robot_python="${DEPLOY_PYTHON:-/usr/bin/python3}"
"$robot_python" - <<'PYTHON'
import os, sys
if os.environ.get('ROS_DISTRO') != 'foxy':
    raise SystemExit('Source ~/unitree_ros2/setup.sh first (expected ROS 2 Foxy).')
if sys.version_info[:2] != (3, 8):
    raise SystemExit('Foxy on Ubuntu 20.04 requires its system Python 3.8; do not use Conda.')
import rclpy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
PYTHON
if [[ -x "$repo_dir/.venv-deploy/bin/python" ]]; then
  "$repo_dir/.venv-deploy/bin/python" - <<'PYTHON'
import sys
if sys.version_info[:2] != (3, 8):
    raise SystemExit('Existing .venv-deploy uses another Python. Preserve it and choose a fresh deployment directory.')
PYTHON
fi
"$robot_python" -m venv --system-site-packages "$repo_dir/.venv-deploy"
"$repo_dir/.venv-deploy/bin/python" -m pip install --only-binary=:all: -r "$deploy_dir/requirements.txt"
"$repo_dir/.venv-deploy/bin/python" -m pip freeze > "$repo_dir/.venv-deploy/installed.txt"
"$repo_dir/.venv-deploy/bin/python" "$deploy_dir/run.py" --check
echo "Activate: source $repo_dir/.venv-deploy/bin/activate"
echo 'ROS drivers, localization and Unitree SDK are not installed or started by this script.'
