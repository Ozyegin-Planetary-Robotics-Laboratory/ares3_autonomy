# CubeMars AK70 & ZED Camera Autonomous Driving Repository

This repository contains files that control the CubeMars AK70 motor and retrieve sensor data from the ZED Camera, processing them for autonomous driving tasks.

## Overview

### `loco_lib.py`
- Converts joystick inputs to velocity commands  
- Sends CAN messages to motors

### `zed_autonomy_v3.py`
- Implements ArUco marker following  
- Contains obstacle detection algorithms

### `getGps.py`
- Reads CAN data from the Here4 GPS module

### `gps_lib.py`
- Provides functions for coordinate system calculations
