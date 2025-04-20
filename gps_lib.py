import math, json
import numpy as np

def calculate_vector(lat1, lon1, lat2, lon2, unit='m'):
    """
    Calculate the vector between two geographic coordinates.
    
    Parameters:
    lat1, lon1: Latitude and longitude of the first point (in degrees)
    lat2, lon2: Latitude and longitude of the second point (in degrees)
    unit: Unit of distance ('km' for kilometers, 'm' for meters, 'mi' for miles)
    
    Returns:
    A tuple containing:
    - distance: The magnitude of the vector (distance between points)
    - bearing: The direction of the vector (initial bearing in degrees from North)
    - vector_components: (x, y) components of the vector where:
        x: East-West component (positive is East)
        y: North-South component (positive is North)
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Earth radius in different units
    if unit == 'km':
        R = 6371.0  # Earth radius in kilometers
    elif unit == 'm':
        R = 6371000.0  # Earth radius in meters
    elif unit == 'mi':
        R = 3958.8  # Earth radius in miles
    else:
        raise ValueError("Unit must be 'km', 'm', or 'mi'")
    
    # Calculate differences
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula for distance
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    # Calculate bearing (direction)
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    bearing = (math.degrees(bearing_rad) + 360) % 360  # Convert to degrees and normalize
    
    # Calculate vector components
    x_component = distance * math.sin(math.radians(bearing))  # East-West component
    y_component = distance * math.cos(math.radians(bearing))  # North-South component
    
    return distance, bearing, (x_component, y_component)


def ll2meters(lat1, lon1, lat2, lon2):
    R=6378.137
    dLat = lat2*math.pi/180 - lat1*math.pi/180
    dLon = lon2*math.pi/180 - lon1*math.pi/180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1*math.pi/180)*math.cos(lat2*math.pi/10)*math.sin(dLon/2)*math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    return d*1000

def ll2bearing(lat1, lon1, lat2, lon2):
    R=6378.137
    dLat = lat2*math.pi/180 - lat1*math.pi/180
    dLon = lon2*math.pi/180 - lon1*math.pi/180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1*math.pi/180)*math.cos(lat2*math.pi/10)*math.sin(dLon/2)*math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c

    return math.atan2(math.sin(dLon)*math.cos(lat2),math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dLon))

def add_vector_to_coordinates(lat, lon, distance, bearing, unit='m'):
    """
    Calculate the destination point given a starting point, distance, and bearing.
    
    Parameters:
    lat, lon: Latitude and longitude of the starting point (in degrees)
    distance: Distance to travel (in the specified unit)
    bearing: Direction to travel (in degrees from North, clockwise)
    unit: Unit of distance ('km' for kilometers, 'm' for meters, 'mi' for miles)
    
    Returns:
    A tuple containing the latitude and longitude of the destination point (in degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    # Earth radius in different units
    if unit == 'km':
        R = 6371.0  # Earth radius in kilometers
    elif unit == 'm':
        R = 6371000.0  # Earth radius in meters
    elif unit == 'mi':
        R = 3958.8  # Earth radius in miles
    else:
        raise ValueError("Unit must be 'km', 'm', or 'mi'")
    
    # Calculate the angular distance in radians
    angular_distance = distance / R
    
    # Calculate the destination point
    lat2_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    
    lon2_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(lat2_rad)
    )
    
    # Normalize longitude to -180 to +180 degrees
    lon2_rad = ((lon2_rad + 3 * math.pi) % (2 * math.pi)) - math.pi
    
    # Convert back to degrees
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)
    
    return lat2, lon2

def add_vector_components_to_coordinates(lat, lon, x_component, y_component, unit='m'):
    """
    Calculate the destination point given a starting point and vector components.
    
    Parameters:
    lat, lon: Latitude and longitude of the starting point (in degrees)
    x_component: East-West component of the vector (positive is East)
    y_component: North-South component of the vector (positive is North)
    unit: Unit of distance ('km' for kilometers, 'm' for meters, 'mi' for miles)
    
    Returns:
    A tuple containing the latitude and longitude of the destination point (in degrees)
    """
    # Calculate the distance and bearing from the vector components
    distance = math.sqrt(x_component**2 + y_component**2)
    bearing = (math.degrees(math.atan2(x_component, y_component)) + 360) % 360
    
    # Use the existing function to calculate the destination
    lat = float(f"{lat:.8f}")
    lon = float(f"{lon:.8f}")
    return add_vector_to_coordinates(lat, lon, distance, bearing, unit)



def ll2meters(lat1, lon1, lat2, lon2):
    R=6378.137
    dLat = lat2*math.pi/180 - lat1*math.pi/180
    dLon = lon2*math.pi/180 - lon1*math.pi/180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1*math.pi/180)*math.cos(lat2*math.pi/10)*math.sin(dLon/2)*math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    return d*1000

def ll2bearing(lat1, lon1, lat2, lon2):
    R=6378.137
    dLat = lat2*math.pi/180 - lat1*math.pi/180
    dLon = lon2*math.pi/180 - lon1*math.pi/180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1*math.pi/180)*math.cos(lat2*math.pi/10)*math.sin(dLon/2)*math.sin(dLon/2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c

    return math.atan2(math.sin(dLon)*math.cos(lat2),math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dLon))

def read_gps():
    d=json.loads(open(".gps.json","r").read())
    return d