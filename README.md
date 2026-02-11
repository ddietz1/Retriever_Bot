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
