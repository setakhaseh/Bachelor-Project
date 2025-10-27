import carla
import random
import time
import numpy as np
import cv2

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get blueprints and spawn points
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Spawn a vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
if vehicle is None:
    raise RuntimeError("Failed to spawn vehicle!")

# Attach spectator to follow the car
spectator = world.get_spectator()
transform = carla.Transform(
    vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
    vehicle.get_transform().rotation
)
spectator.set_transform(transform)

# Create and attach RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Shared dictionary for image data
camera_data = {'image': np.zeros((600, 800, 4), dtype=np.uint8)}

# Callback for camera images
def camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    data_dict['image'] = array

camera.listen(lambda image: camera_callback(image, camera_data))

# Enable autopilot
vehicle.set_autopilot(True)

# Display loop
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
try:
    while True:
        img = camera_data['image']
        cv2.imshow('RGB Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    camera.stop()
    vehicle.destroy()
