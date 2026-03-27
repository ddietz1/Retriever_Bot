# Retriever_Bot
## Summary
This project implements an autonomous underwater object retrieval system using a BlueROV2 platform running ArduSub, BlueOS, and ROS2. The goal is to detect, navigate to, and retrieve a submerged ring using onboard vision and closed-loop control.

## Overview
The system integrates:

* ROS2 nodes for perception, control, and MAVROS bridging

* Real-time object detection for ring localization

* Velocity-based control via MAVROS setpoint commands

* Newton Subsea Gripper actuation for object retrieval

* Hardware-level integration with Pixhawk and ArduSub

* The autonomy pipeline follows this structure:

* Camera stream → object detection node

* Ring pose estimation → control node

* Velocity commands → MAVROS → ArduSub

* Gripper actuation when within capture threshold

* The project demonstrates full-stack robotics integration, including:

* Underwater vehicle control

* Vision-based servoing

* ROS2 system architecture

* MAVLink/MAVROS communication

* Hardware-in-the-loop testing

This repository contains the ROS2 packages, control logic, perception pipeline, and system architecture used to achieve autonomous underwater ring retrieval.

## Installation
1. Clone the repository into your ROS 2 workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/ddietz1/Retriever_Bot.git
```
2. Install Python dependencies
```bash
pip install transformers
```
3. Install Python dependencies
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```
4. Build the workspace
```bash
cd ~/ros2_ws
colcon build --packages-select bluerov_control
source install/setup.bash
```

## Running the system
```bash
ros2 launch bluerov_control retriever_launch.py
```
This starts:
* The MAVROS bridge
* The camera/perception pipeline
* The control node

For full object detection and retrieval:
```bash
ros2 service call /bluerov/detect std_srvs/srv/Empty
```

## Package architecture
```
Retriever_Bot/
├── bluerov_control/          # Main ROS 2 package
|   ├── bridge_node.py        # MAVROS bridge
|   ├── camera_node.py        # Retreives camera info from ROV
│   ├── object_detection.py   # Vision pipeline & ring localization
│   ├── control.py            # Velocity-based servoing & gripper logic
|   ├── yolo_detect.py        # Runs image topics through yolo model
│   └── ...
├── bluerov_heading/          # ROS 2 package for heading control
|   ├── heading_node.py       # Determines current heading of the ROV
├── launch/                   # ROS2 launch files
├── resource/                 # Package resource files
├── package.xml               # ROS2 package manifest
├── setup.py                  # Python package setup
└── setup.cfg
```
