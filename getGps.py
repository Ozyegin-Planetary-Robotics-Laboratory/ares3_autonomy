#!/usr/bin/env python3
import dronecan
import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Here4 GNSS position monitor')
    parser.add_argument('--interface', '-i', default='can1', help='CAN interface name')
    parser.add_argument('--bitrate', '-b', default=1000000, type=int, help='CAN bus bitrate')
    args = parser.parse_args()

    # Initialize DroneCAN node
    node = dronecan.make_node(args.interface, 
                             bitrate=args.bitrate,
                             bustype='socketcan')
    
    # Set up handler for GNSS Fix messages
    node.add_handler(dronecan.uavcan.equipment.gnss.Fix, on_gnss_fix)
    node.add_handler(dronecan.uavcan.equipment.gnss.Fix2, on_gnss_fix2)
    
    #print(f"Monitoring Here4 GNSS position on {args.interface}...")
    #print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Process received messages
            node.spin(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")

def on_gnss_fix(msg):
    """Handler for GNSS Fix messages"""
    try:
        # Extract position data
        lat_deg = msg.message.latitude_deg_1e8 * 1e-8
        lon_deg = msg.message.longitude_deg_1e8 * 1e-8
        
        # Get fix status if available
        fix_ok = True
        if hasattr(msg.message, 'status_flags'):
            status_flags = msg.message.status_flags
            fix_ok = bool(status_flags & 1)
        
        if fix_ok:
            print(f"Position: Lat: {lat_deg:.8f}, Lon: {lon_deg:.8f}")
    except Exception as e:
        # Just log the error and continue
        print(f"Error in Fix handler: {e}")

def on_gnss_fix2(msg):
    """Handler for GNSS Fix2 messages (extended version)"""
    try:
        # Extract position data
        lat_deg = msg.message.latitude_deg_1e8 * 1e-8
        lon_deg = msg.message.longitude_deg_1e8 * 1e-8
        
        # Different versions of DroneCAN might have different field names
        # Try different possible field names for status flags
        fix_ok = True
        
        # Try different possible field names for status
        if hasattr(msg.message, 'status_flags'):
            status_flags = msg.message.status_flags
            fix_ok = bool(status_flags & 1)
        elif hasattr(msg.message, 'status'):
            status = msg.message.status
            fix_ok = bool(status & 1)
        elif hasattr(msg.message, 'gnss_status'):
            status = msg.message.gnss_status
            fix_ok = bool(status & 1)
        
        # Always print position for Fix2 messages (they're usually valid)
        # You can remove this condition if you only want valid fixes
        if fix_ok:
            # print(f"Position: Lat: {lat_deg:.7f}, Lon: {lon_deg:.7f}")
            with open(".gps.json","w") as f:
                json.dump({"lat":lat_deg,"lon":lon_deg},f)
    except Exception as e:
        pass
        # Just log the error and continue
        # print(f"Error in Fix2 handler: {e}")

def get_gps():
    # lat_deg = msg.message.latitude_deg_1e8 * 1e-8
    # lon_deg = msg.message.longitude_deg_1e8 * 1e-8
    return 41.0309088, 29.2588675

if __name__ == "__main__":
    main()
