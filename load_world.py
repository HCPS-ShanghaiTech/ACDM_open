"""
Load map and view
"""

import time
import carla

# Connect to the CARLA server
start = time.time()
client = carla.Client("localhost", 2000)
client.set_timeout(100.0)  # Set timeout
# Load a new map
client.load_world("Town05")


world = client.get_world()
transform = carla.Transform()
# Set the spectator
spectator = world.get_spectator()
bv_transform = carla.Transform(
    transform.location + carla.Location(z=250, x=0),
    carla.Rotation(yaw=0, pitch=-90),
)
spectator.set_transform(bv_transform)
