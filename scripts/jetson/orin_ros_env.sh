# source this on the ROSMASTER M3 Pro Orin (any user) before talking to the board:
#   source ~/github/physical-ai-jetson-robotics/scripts/jetson/orin_ros_env.sh
export ROS_DOMAIN_ID=30                      # vendor firmware + agent domain
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/github/physical-ai-jetson-robotics/scripts/jetson/config/fastdds_no_shm.xml"
source /opt/ros/humble/setup.bash
[ -f /home/jetson/yahboomcar_ws/install/setup.bash ] && source /home/jetson/yahboomcar_ws/install/setup.bash   # arm_msgs etc.
[ -f /home/jetson/M3Pro_ws/install/setup.bash ] && source /home/jetson/M3Pro_ws/install/setup.bash 2>/dev/null
# preflight: the board must be discoverable, else nothing below is valid
m3pro_preflight() { timeout 20 ros2 node list --no-daemon --spin-time 5 2>/dev/null | grep -q '^/YB_Node$' && echo "[m3pro] board OK (/YB_Node)" || { echo "[m3pro] board NOT visible — restart the micro-ROS agent (see docs/notes/yahboom_strategy_2026-08-20.md s.9)"; return 1; }; }
